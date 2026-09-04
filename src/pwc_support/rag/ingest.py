from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class CorpusDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    path: str
    title: str
    url: HttpUrl | None = None
    language: str = "en"
    source_status: str = "active"
    version: str = "1"
    checksum: str | None = None
    text: str = ""

    @field_validator("path")
    @classmethod
    def reject_unsafe_path(cls, value: str) -> str:
        """Corpus paths must stay inside the corpus root: no absolutes, no traversal."""
        candidate = PurePosixPath(value)
        if candidate.is_absolute() or ".." in candidate.parts or not value.strip():
            raise ValueError(f"unsafe corpus path: {value!r}")
        return value


class ChunkingConfig(BaseModel):
    chunk_size_tokens: int = Field(default=300, ge=20, le=2000)
    chunk_overlap_tokens: int = Field(default=50, ge=0, le=500)

    @model_validator(mode="after")
    def validate_overlap(self) -> ChunkingConfig:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk overlap must be smaller than chunk size")
        return self


class CorpusChunk(BaseModel):
    chunk_id: str
    source_id: str
    title: str
    heading: str
    original_text: str
    context: str
    embedding_text: str
    token_count: int
    chunk_index: int
    document_version: str
    document_checksum: str
    canonical_url: HttpUrl | None = None
    language: str = "en"
    source_status: str = "active"


class ChunkContextualizer(Protocol):
    def contextualize(self, *, document: CorpusDocument, heading: str, chunk: str) -> str: ...


class MetadataContextualizer:
    def contextualize(self, *, document: CorpusDocument, heading: str, chunk: str) -> str:
        return (
            f"This chunk is from {document.title}, section {heading}, "
            f"source version {document.version}."
        )


class TextGenerator(Protocol):
    def text(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str: ...


class ModelContextualizer:
    def __init__(self, generator: TextGenerator, max_document_chars: int = 24000) -> None:
        self.generator = generator
        self.max_document_chars = max_document_chars

    def contextualize(self, *, document: CorpusDocument, heading: str, chunk: str) -> str:
        return self.generator.text(
            system="Write a short retrieval context using only the supplied retail document.",
            user=(
                f"<document>{document.text[:self.max_document_chars]}</document>"
                f"<heading>{heading}</heading><chunk>{chunk}</chunk>"
            ),
            max_tokens=100,
            temperature=0.0,
        )



class ManifestError(ValueError):
    """The corpus manifest is not a usable description of the knowledge base."""


def load_manifest(path: Path) -> tuple[CorpusDocument, ...]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    documents = tuple(CorpusDocument.model_validate(item) for item in payload["documents"])
    _reject_duplicates("source_id", [document.source_id for document in documents])
    _reject_duplicates("path", [document.path for document in documents])
    return documents


def _reject_duplicates(field: str, values: list[str]) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ManifestError(f"duplicate {field} in manifest: {', '.join(duplicates)}")


def load_documents(root: Path, manifest: tuple[CorpusDocument, ...]) -> tuple[CorpusDocument, ...]:
    """Read every declared document, verifying any checksum the manifest commits to."""
    resolved_root = root.resolve()
    loaded: list[CorpusDocument] = []
    for document in manifest:
        source = (resolved_root / document.path).resolve()
        if not source.is_relative_to(resolved_root):
            raise ManifestError(f"corpus path escapes the corpus root: {document.path}")
        if not source.is_file():
            raise ManifestError(f"missing corpus document: {document.path}")
        text = source.read_text(encoding="utf-8")
        checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if document.checksum and document.checksum != checksum:
            raise ManifestError(
                f"checksum mismatch for {document.source_id}: "
                f"declared {document.checksum}, found {checksum}"
            )
        loaded.append(document.model_copy(update={"text": text}))
    return tuple(loaded)


def chunk_document(
    document: CorpusDocument,
    config: ChunkingConfig | None = None,
    contextualizer: ChunkContextualizer | None = None,
) -> tuple[CorpusChunk, ...]:
    resolved = config or ChunkingConfig()
    context_provider = contextualizer or MetadataContextualizer()
    checksum = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
    sections = _markdown_sections(document.text, document.title)
    chunks: list[CorpusChunk] = []
    index = 0
    for heading, section in sections:
        words = section.split()
        step = resolved.chunk_size_tokens - resolved.chunk_overlap_tokens
        for start in range(0, len(words), step):
            window = words[start : start + resolved.chunk_size_tokens]
            if not window:
                continue
            original = " ".join(window)
            context = context_provider.contextualize(
                document=document, heading=heading, chunk=original
            )
            payload = "\x1f".join(
                (document.source_id, document.version, heading, str(index), context, original)
            )
            chunk_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            chunks.append(
                CorpusChunk(
                    chunk_id=chunk_id,
                    source_id=document.source_id,
                    title=document.title,
                    heading=heading,
                    original_text=original,
                    context=context,
                    embedding_text=f"{context}\n\n{original}",
                    token_count=len(window),
                    chunk_index=index,
                    document_version=document.version,
                    document_checksum=checksum,
                    canonical_url=document.url,
                    language=document.language,
                    source_status=document.source_status,
                )
            )
            index += 1
            if start + resolved.chunk_size_tokens >= len(words):
                break
    return tuple(chunks)


def prepare_chunks(
    documents: tuple[CorpusDocument, ...],
    config: ChunkingConfig | None = None,
    contextualizer: ChunkContextualizer | None = None,
) -> tuple[CorpusChunk, ...]:
    return tuple(
        chunk
        for document in documents
        for chunk in chunk_document(document, config, contextualizer)
    )


def _markdown_sections(text: str, default_heading: str) -> list[tuple[str, str]]:
    heading = default_heading
    body: list[str] = []
    sections: list[tuple[str, str]] = []
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+)$", line.strip())
        if match:
            if body:
                sections.append((heading, "\n".join(body).strip()))
                body = []
            heading = match.group(1).strip()
        elif line.strip():
            body.append(line.strip())
    if body:
        sections.append((heading, "\n".join(body).strip()))
    return [(section_heading, section) for section_heading, section in sections if section]

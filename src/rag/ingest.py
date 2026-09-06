from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
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
        system = "Write a short retrieval context using only the supplied retail document."
        user = (
            f"<document>{document.text[: self.max_document_chars]}</document>"
            f"<heading>{heading}</heading><chunk>{chunk}</chunk>"
        )
        if hasattr(self.generator, "invoke"):
            res = self.generator.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            return (res.content if isinstance(res.content, str) else str(res.content)).strip()
        return self.generator.text(
            system=system,
            user=user,
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


MARKDOWN_HEADERS = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]


def _split_into_sections(document: CorpusDocument) -> list[tuple[str, str]]:
    path = PurePosixPath(document.path)
    ext = path.suffix.lower()
    text = document.text.strip()
    if not text:
        return []

    if ext in {".md", ".markdown"}:
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=MARKDOWN_HEADERS,
            strip_headers=True,
        )
        splits = header_splitter.split_text(text)
        sections: list[tuple[str, str]] = []
        for split in splits:
            content = split.page_content.strip()
            if not content:
                continue
            meta = split.metadata
            heading = (
                meta.get("Header 4")
                or meta.get("Header 3")
                or meta.get("Header 2")
                or meta.get("Header 1")
                or document.title
            )
            sections.append((heading, content))
        if sections:
            return sections

    return [(document.title, text)]


def _build_sub_splitter(path_str: str, config: ChunkingConfig) -> RecursiveCharacterTextSplitter:
    ext = PurePosixPath(path_str).suffix.lower()

    def length_fn(text: str) -> int:
        return len(text.split())

    if ext in {".md", ".markdown"}:
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=config.chunk_size_tokens,
            chunk_overlap=config.chunk_overlap_tokens,
            length_function=length_fn,
        )
    if ext in {".html", ".htm"}:
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.HTML,
            chunk_size=config.chunk_size_tokens,
            chunk_overlap=config.chunk_overlap_tokens,
            length_function=length_fn,
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size_tokens,
        chunk_overlap=config.chunk_overlap_tokens,
        length_function=length_fn,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_document(
    document: CorpusDocument,
    config: ChunkingConfig | None = None,
    contextualizer: ChunkContextualizer | None = None,
) -> tuple[CorpusChunk, ...]:
    resolved = config or ChunkingConfig()
    context_provider = contextualizer or MetadataContextualizer()
    checksum = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
    sections = _split_into_sections(document)
    sub_splitter = _build_sub_splitter(document.path, resolved)
    chunks: list[CorpusChunk] = []
    index = 0
    for heading, section_text in sections:
        sub_chunks = sub_splitter.split_text(section_text)
        for sub_chunk in sub_chunks:
            original = sub_chunk.strip()
            if not original:
                continue
            context = context_provider.contextualize(
                document=document, heading=heading, chunk=original
            )
            payload = "\x1f".join(
                (document.source_id, document.version, heading, str(index), context, original)
            )
            chunk_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            token_count = len(original.split())
            chunks.append(
                CorpusChunk(
                    chunk_id=chunk_id,
                    source_id=document.source_id,
                    title=document.title,
                    heading=heading,
                    original_text=original,
                    context=context,
                    embedding_text=f"{context}\n\n{original}",
                    token_count=token_count,
                    chunk_index=index,
                    document_version=document.version,
                    document_checksum=checksum,
                    canonical_url=document.url,
                    language=document.language,
                    source_status=document.source_status,
                )
            )
            index += 1
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

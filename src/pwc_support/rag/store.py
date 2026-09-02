from __future__ import annotations

from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel

from pwc_support.domain.models import RagRequest, RetrievalBatch, RetrievalHit
from pwc_support.rag.ingest import CorpusChunk
from pwc_support.rag.lexical import LexicalIndex


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SyncReport(BaseModel):
    upserted: int = 0
    unchanged: int = 0
    deleted: int = 0


class ChromaKnowledgeBase:
    """Chroma adapter with explicit source metadata and deterministic filters."""

    def __init__(
        self,
        client: Any,
        collection_name: str,
        embedder: Embedder,
        lexical_index: LexicalIndex | None = None,
    ) -> None:
        self.collection = client.get_or_create_collection(collection_name)
        self.embedder = embedder
        self.lexical_index = lexical_index

    def sync(
        self,
        chunks: tuple[CorpusChunk, ...],
        *,
        batch_size: int = 16,
        managed_source_ids: set[str] | None = None,
    ) -> SyncReport:
        source_ids = managed_source_ids or {chunk.source_id for chunk in chunks}
        current = {
            chunk.chunk_id: chunk for chunk in chunks if chunk.source_id in source_ids
        }
        existing_ids: set[str] = set()
        for source_id in source_ids:
            existing = self.collection.get(where={"source_id": source_id})
            existing_ids.update(str(item) for item in existing.get("ids", []))
        stale_ids = existing_ids - set(current)
        if stale_ids:
            self.collection.delete(ids=sorted(stale_ids))
        changed = [chunk for chunk_id, chunk in current.items() if chunk_id not in existing_ids]
        for start in range(0, len(changed), batch_size):
            batch = changed[start : start + batch_size]
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.original_text for chunk in batch],
                embeddings=self.embedder.embed([chunk.embedding_text for chunk in batch]),
                metadatas=[self._metadata(chunk) for chunk in batch],
            )
        if self.lexical_index is not None:
            self.lexical_index.sync(chunks, source_ids)
        return SyncReport(
            upserted=len(changed), unchanged=len(current) - len(changed), deleted=len(stale_ids)
        )

    def delete_source(self, source_id: str) -> int:
        existing = self.collection.get(where={"source_id": source_id})
        ids = [str(item) for item in existing.get("ids", [])]
        if ids:
            self.collection.delete(ids=ids)
        if self.lexical_index is not None:
            self.lexical_index.delete_source(source_id)
        return len(ids)

    def count(self) -> int:
        return int(self.collection.count())

    @staticmethod
    def _metadata(chunk: CorpusChunk) -> dict[str, str | int]:
        return {
            "source_id": chunk.source_id,
            "title": chunk.title,
            "canonical_url": str(chunk.canonical_url) if chunk.canonical_url else "",
            "language": chunk.language,
            "source_status": chunk.source_status,
            "heading": chunk.heading,
            "context": chunk.context,
            "document_version": chunk.document_version,
            "document_checksum": chunk.document_checksum,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
        }

    def retrieve(
        self, request: RagRequest | str, *, language: str = "en", top_k: int = 6
    ) -> RetrievalBatch:
        if isinstance(request, str):
            request = RagRequest(
                question=request,
                language=cast(Literal["en"], language),
            )
        where = {"$and": [{"language": request.language}, {"source_status": "active"}]}
        result = self.collection.query(
            query_embeddings=self.embedder.embed([request.question]),
            n_results=top_k,
            where=where,
        )
        distances = (result.get("distances") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        vector_hits = tuple(
            RetrievalHit(
                source_id=str(metadata["source_id"]),
                chunk_id=str(result["ids"][0][index]),
                title=str(metadata["title"]),
                text=str(documents[index]),
                heading=str(metadata.get("heading", "")),
                canonical_url=metadata.get("canonical_url") or None,
                similarity=max(-1.0, min(1.0, 1.0 - float(distances[index]))),
                language=str(metadata["language"]),
                source_status=str(metadata["source_status"]),
            )
            for index, metadata in enumerate(metadatas)
        )
        if self.lexical_index is None:
            return RetrievalBatch(hits=vector_hits)
        lexical_ids = self.lexical_index.search(request.question, limit=top_k * 3)
        if not lexical_ids:
            return RetrievalBatch(hits=vector_hits)
        lexical_result = self.collection.get(ids=lexical_ids, where=where)
        lexical_hits = self._hits_from_get(lexical_result)
        return RetrievalBatch(hits=self._fuse(vector_hits, lexical_hits, top_k))

    @staticmethod
    def _hits_from_get(result: dict[str, Any]) -> tuple[RetrievalHit, ...]:
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return tuple(
            RetrievalHit(
                source_id=str(metadata["source_id"]), chunk_id=str(result["ids"][index]),
                title=str(metadata["title"]), text=str(documents[index]),
                heading=str(metadata.get("heading", "")),
                canonical_url=metadata.get("canonical_url") or None, similarity=0.0,
                language=str(metadata["language"]), source_status=str(metadata["source_status"]),
                source_type="lexical",
            )
            for index, metadata in enumerate(metadatas)
        )

    @staticmethod
    def _fuse(
        vector_hits: tuple[RetrievalHit, ...],
        lexical_hits: tuple[RetrievalHit, ...],
        top_k: int,
    ) -> tuple[RetrievalHit, ...]:
        scores: dict[str, float] = {}
        by_id = {hit.chunk_id: hit for hit in lexical_hits}
        by_id.update({hit.chunk_id: hit for hit in vector_hits})
        for ranking in (vector_hits, lexical_hits):
            for rank, hit in enumerate(ranking, start=1):
                scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (60 + rank)
        ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:top_k]
        return tuple(by_id[chunk_id] for chunk_id in ordered)

from __future__ import annotations

import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel

from domain.models import RagRequest, RetrievalBatch, RetrievalHit
from rag.ingest import CorpusChunk


class LexicalIndex:
    """Persistent FTS5 index using SQLite's BM25 ranking function."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
                "chunk_id UNINDEXED, source_id UNINDEXED, content)"
            )

    def sync(self, chunks: tuple[CorpusChunk, ...], source_ids: set[str]) -> None:
        with self._connect() as connection:
            for source_id in source_ids:
                connection.execute("DELETE FROM chunks_fts WHERE source_id = ?", (source_id,))
            connection.executemany(
                "INSERT INTO chunks_fts(chunk_id, source_id, content) VALUES (?, ?, ?)",
                [
                    (chunk.chunk_id, chunk.source_id, chunk.embedding_text)
                    for chunk in chunks
                    if chunk.source_id in source_ids
                ],
            )

    def search(self, query: str, *, limit: int = 20) -> list[str]:
        terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]*", query)
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY bm25(chunks_fts) LIMIT ?",
                (expression, limit),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def delete_source(self, source_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM chunks_fts WHERE source_id = ?", (source_id,))

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


def chroma_client(settings: Any) -> Any:
    """Resolve the configured Chroma backend so ingestion and serving always agree."""
    import chromadb

    if settings.chroma_mode == "http":
        return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    return chromadb.PersistentClient(path=str(settings.chroma_path))


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
        top_k: int = 6,
    ) -> None:
        # Cosine space is required: these are unnormalised nomic-embed-text vectors, and
        # the retrieval threshold is expressed as a cosine similarity of 1 - distance.
        self.collection = client.get_or_create_collection(
            collection_name, configuration={"hnsw": {"space": "cosine"}}
        )
        self.embedder: Any = embedder
        self.lexical_index = lexical_index
        self.top_k = top_k

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        fn = (
            getattr(self.embedder, "embed_documents", None)
            or getattr(self.embedder, "embed", None)
        )
        if fn is not None:
            return [[float(v) for v in vec] for vec in fn(texts)]
        raise ValueError("Configured embedder does not support embed_documents or embed")

    def _embed_query(self, text: str) -> list[float]:
        fn_query = getattr(self.embedder, "embed_query", None)
        if fn_query is not None:
            return [float(v) for v in fn_query(text)]
        fn_docs = (
            getattr(self.embedder, "embed_documents", None)
            or getattr(self.embedder, "embed", None)
        )
        if fn_docs is not None:
            return [float(v) for v in fn_docs([text])[0]]
        raise ValueError("Configured embedder does not support embed_query or embed")

    def sync(
        self,
        chunks: tuple[CorpusChunk, ...],
        *,
        batch_size: int = 16,
        managed_source_ids: set[str] | None = None,
        full_reconciliation: bool = False,
    ) -> SyncReport:
        manifest_source_ids = {chunk.source_id for chunk in chunks}
        source_ids = set(managed_source_ids or manifest_source_ids)
        if full_reconciliation:
            existing_metas = self.collection.get(include=["metadatas"]).get("metadatas", [])
            for m in existing_metas:
                if m and "source_id" in m:
                    source_ids.add(str(m["source_id"]))

        current = {chunk.chunk_id: chunk for chunk in chunks if chunk.source_id in source_ids}
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
                embeddings=self._embed_documents([chunk.embedding_text for chunk in batch]),
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
        self, request: RagRequest | str, *, language: str = "en", top_k: int | None = None
    ) -> RetrievalBatch:
        top_k = self.top_k if top_k is None else top_k
        if isinstance(request, str):
            request = RagRequest(
                question=request,
                language=cast(Literal["en"], language),
            )
        where = {"$and": [{"language": request.language}, {"source_status": "active"}]}
        query_embedding = self._embed_query(request.question)
        result = self.collection.query(
            query_embeddings=[query_embedding],
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
        lexical_result = self.collection.get(
            ids=lexical_ids,
            where=where,
            include=["documents", "metadatas", "embeddings"],
        )
        lexical_hits = self._hits_from_get(lexical_result, query_embedding)
        return RetrievalBatch(hits=self._fuse(vector_hits, lexical_hits, top_k))

    @classmethod
    def _hits_from_get(
        cls, result: dict[str, Any], query_embedding: list[float]
    ) -> tuple[RetrievalHit, ...]:
        """Score keyword hits against the same query vector so one threshold governs both."""
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        embeddings = result.get("embeddings")
        return tuple(
            RetrievalHit(
                source_id=str(metadata["source_id"]),
                chunk_id=str(result["ids"][index]),
                title=str(metadata["title"]),
                text=str(documents[index]),
                heading=str(metadata.get("heading", "")),
                canonical_url=metadata.get("canonical_url") or None,
                similarity=cls._cosine(
                    query_embedding,
                    None if embeddings is None else embeddings[index],
                ),
                language=str(metadata["language"]),
                source_status=str(metadata["source_status"]),
                source_type="lexical",
            )
            for index, metadata in enumerate(metadatas)
        )

    @staticmethod
    def _cosine(left: list[float], right: Any) -> float:
        if right is None or len(right) != len(left):
            return 0.0
        dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
        norm = math.sqrt(sum(float(a) ** 2 for a in left)) * math.sqrt(
            sum(float(b) ** 2 for b in right)
        )
        return 0.0 if norm == 0.0 else max(-1.0, min(1.0, dot / norm))

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

from __future__ import annotations

from typing import Any, Literal, Protocol, cast

from pwc_support.domain.models import RagRequest, RetrievalBatch, RetrievalHit
from pwc_support.rag.ingest import CorpusDocument


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class ChromaKnowledgeBase:
    """Chroma adapter with explicit source metadata and deterministic filters."""

    def __init__(self, client: Any, collection_name: str, embedder: Embedder) -> None:
        self.collection = client.get_or_create_collection(collection_name)
        self.embedder = embedder

    def ingest(self, documents: tuple[CorpusDocument, ...]) -> None:
        ids = [document.source_id for document in documents]
        self.collection.upsert(
            ids=ids,
            documents=[document.text for document in documents],
            embeddings=self.embedder.embed([document.text for document in documents]),
            metadatas=[
                {
                    "source_id": document.source_id,
                    "title": document.title,
                    "canonical_url": str(document.url) if document.url else "",
                    "language": document.language,
                    "source_status": document.source_status,
                    "heading": document.title,
                }
                for document in documents
            ],
        )

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
        hits = tuple(
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
        return RetrievalBatch(hits=hits)

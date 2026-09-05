from pathlib import Path

import chromadb

from rag.ingest import (
    ChunkingConfig,
    load_documents,
    load_manifest,
    prepare_chunks,
)
from rag.store import ChromaKnowledgeBase


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        return [[1.0, 0.0] if "shipping" in text.lower() else [0.0, 1.0] for text in texts]


def test_chroma_store_retrieves_attributed_hits_with_language_filter() -> None:
    root = Path(__file__).parents[3] / "corpus"
    embedder = FakeEmbedder()
    store = ChromaKnowledgeBase(chromadb.EphemeralClient(), "test_collection", embedder)
    chunks = prepare_chunks(
        load_documents(root, load_manifest(root / "manifest.json")),
        ChunkingConfig(chunk_size_tokens=30, chunk_overlap_tokens=5),
    )
    report = store.sync(chunks, batch_size=2)

    batch = store.retrieve("shipping support", language="en", top_k=2)

    assert report.upserted == len(chunks)
    assert max(embedder.calls[:-1]) <= 2
    assert batch.hits
    assert batch.hits[0].source_id == "shipping-and-orders"
    assert batch.hits[0].language == "en"


def test_sync_removes_stale_chunks_and_source_delete_is_explicit() -> None:
    root = Path(__file__).parents[3] / "corpus"
    store = ChromaKnowledgeBase(chromadb.EphemeralClient(), "sync_collection", FakeEmbedder())
    chunks = prepare_chunks(load_documents(root, load_manifest(root / "manifest.json")))
    store.sync(chunks)

    reduced = tuple(chunk for chunk in chunks if chunk.source_id != "warranty-and-support")
    report = store.sync(reduced, managed_source_ids={"warranty-and-support"})

    assert report.deleted > 0
    assert store.delete_source("shipping-and-orders") > 0

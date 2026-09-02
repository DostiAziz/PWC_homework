from pathlib import Path

import chromadb

from pwc_support.rag.ingest import load_documents, load_manifest
from pwc_support.rag.store import ChromaKnowledgeBase


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "financial" in text.lower() else [0.0, 1.0] for text in texts]


def test_chroma_store_retrieves_attributed_hits_with_language_filter() -> None:
    root = Path(__file__).parents[3] / "corpus"
    store = ChromaKnowledgeBase(chromadb.EphemeralClient(), "test_collection", FakeEmbedder())
    store.ingest(load_documents(root, load_manifest(root / "manifest.json")))

    batch = store.retrieve("financial services", language="en", top_k=2)

    assert batch.hits
    assert batch.hits[0].source_id == "pwc-financial-services"
    assert batch.hits[0].language == "en"

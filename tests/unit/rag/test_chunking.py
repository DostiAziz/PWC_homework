from pathlib import Path

from rag.ingest import (
    ChunkingConfig,
    CorpusDocument,
    chunk_document,
    load_documents,
    load_manifest,
)


class FakeContextualizer:
    def contextualize(self, *, document: CorpusDocument, heading: str, chunk: str) -> str:
        return f"Context for {document.source_id} and {heading}."


def test_manifest_loads_curated_documents_and_preserves_attribution() -> None:
    root = Path(__file__).parents[3] / "corpus"

    manifest = load_manifest(root / "manifest.json")
    documents = load_documents(root, manifest)

    assert len(documents) == 3
    assert {item.source_id for item in documents} == {
        "shipping-and-orders",
        "cancellation-policy",
        "warranty-and-support",
    }
    assert all(item.language == "en" for item in documents)
    assert any(item.source_id == "shipping-and-orders" for item in documents)


def test_chunking_is_bounded_overlapping_and_deterministic() -> None:
    root = Path(__file__).parents[3] / "corpus"
    document = load_documents(root, load_manifest(root / "manifest.json"))[0]
    config = ChunkingConfig(chunk_size_tokens=20, chunk_overlap_tokens=5)

    first = chunk_document(document, config)
    second = chunk_document(document, config)

    assert len(first) > 2
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert all(chunk.token_count <= 20 for chunk in first)
    assert set(first[1].original_text.split()[-5:]) <= set(first[2].original_text.split())
    assert all(chunk.heading for chunk in first)


def test_chunk_context_is_prepended_for_embedding_but_original_is_preserved() -> None:
    root = Path(__file__).parents[3] / "corpus"
    document = load_documents(root, load_manifest(root / "manifest.json"))[0]

    chunk = chunk_document(document, contextualizer=FakeContextualizer())[0]

    assert chunk.context.startswith("Context for shipping-and-orders")
    assert chunk.embedding_text == f"{chunk.context}\n\n{chunk.original_text}"

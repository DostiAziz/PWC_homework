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

    assert len(documents) == 4
    assert {item.source_id for item in documents} == {
        "shipping-and-orders",
        "cancellation-policy",
        "warranty-and-support",
        "returns-policy",
    }
    assert all(item.language == "en" for item in documents)
    assert any(item.source_id == "shipping-and-orders" for item in documents)


def test_chunking_is_bounded_and_deterministic() -> None:
    root = Path(__file__).parents[3] / "corpus"
    document = load_documents(root, load_manifest(root / "manifest.json"))[0]
    config = ChunkingConfig(chunk_size_tokens=20, chunk_overlap_tokens=5)

    first = chunk_document(document, config)
    second = chunk_document(document, config)

    assert len(first) > 2
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert all(chunk.token_count <= 20 for chunk in first)
    assert all(chunk.heading for chunk in first)


def test_chunking_overlaps_when_section_exceeds_chunk_size() -> None:
    doc = CorpusDocument(
        source_id="test-doc",
        path="documents/test.md",
        title="Test Policy",
        text=(
            "# Policy Section\n"
            "This is a long section with one two three four five six seven eight nine ten "
            "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen "
            "twenty twentyone twentytwo twentythree twentyfour twentyfive words."
        ),
    )
    config = ChunkingConfig(chunk_size_tokens=20, chunk_overlap_tokens=5)
    chunks = chunk_document(doc, config)

    assert len(chunks) == 2
    assert chunks[0].heading == "Policy Section"
    assert chunks[1].heading == "Policy Section"
    assert all(chunk.token_count <= 20 for chunk in chunks)
    # Check that the 5-token overlap from chunk 0 is present at the start of chunk 1
    assert set(chunks[0].original_text.split()[-5:]) <= set(chunks[1].original_text.split())


def test_chunking_supports_plain_text_documents() -> None:
    doc = CorpusDocument(
        source_id="plain-text-doc",
        path="documents/notes.txt",
        title="General Support Notes",
        text=(
            "First paragraph of notes providing general assistance to customers.\n\n"
            "Second paragraph detailing contact hours and escalations."
        ),
    )
    chunks = chunk_document(doc, ChunkingConfig(chunk_size_tokens=100, chunk_overlap_tokens=10))

    assert len(chunks) >= 1
    assert all(chunk.heading == "General Support Notes" for chunk in chunks)
    assert "First paragraph" in chunks[0].original_text


def test_chunk_context_is_prepended_for_embedding_but_original_is_preserved() -> None:
    root = Path(__file__).parents[3] / "corpus"
    document = load_documents(root, load_manifest(root / "manifest.json"))[0]

    chunk = chunk_document(document, contextualizer=FakeContextualizer())[0]

    assert chunk.context.startswith("Context for shipping-and-orders")
    assert chunk.embedding_text == f"{chunk.context}\n\n{chunk.original_text}"

from pathlib import Path

from pwc_support.rag.ingest import load_documents, load_manifest


def test_manifest_loads_curated_documents_and_preserves_attribution() -> None:
    root = Path(__file__).parents[3] / "corpus"

    manifest = load_manifest(root / "manifest.json")
    documents = load_documents(root, manifest)

    assert len(documents) == 5
    assert {item.source_id for item in documents} == {
        "pwc-global-services", "pwc-financial-services", "pwc-industries",
        "pwc-network-structure", "synthetic-support-faq",
    }
    assert all(item.language == "en" for item in documents)
    assert any(item.source_id == "synthetic-support-faq" for item in documents)

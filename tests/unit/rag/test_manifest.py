import json
from pathlib import Path

import pytest

from retail_support.rag.ingest import ManifestError, load_documents, load_manifest


def _write(path: Path, documents: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"version": "1", "documents": documents}), encoding="utf-8")
    return path


def test_real_manifest_loads_and_every_document_is_present() -> None:
    root = Path(__file__).parents[3] / "corpus"

    documents = load_documents(root, load_manifest(root / "manifest.json"))

    assert len(documents) == 3

    assert all(document.text for document in documents)


def test_duplicate_source_ids_are_rejected(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path / "manifest.json",
        [
            {"source_id": "a", "path": "a.md", "title": "A"},
            {"source_id": "a", "path": "b.md", "title": "B"},
        ],
    )

    with pytest.raises(ManifestError, match="duplicate source_id"):
        load_manifest(manifest)


def test_duplicate_paths_are_rejected(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path / "manifest.json",
        [
            {"source_id": "a", "path": "a.md", "title": "A"},
            {"source_id": "b", "path": "a.md", "title": "B"},
        ],
    )

    with pytest.raises(ManifestError, match="duplicate path"):
        load_manifest(manifest)


@pytest.mark.parametrize("path", ["../secrets.md", "/etc/passwd", "docs/../../escape.md"])
def test_paths_outside_the_corpus_root_are_rejected(tmp_path: Path, path: str) -> None:
    manifest = _write(tmp_path / "manifest.json", [{"source_id": "a", "path": path, "title": "A"}])

    with pytest.raises(ValueError, match="unsafe corpus path"):
        load_manifest(manifest)


def test_declared_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nreal content", encoding="utf-8")
    manifest = _write(
        tmp_path / "manifest.json",
        [{"source_id": "a", "path": "a.md", "title": "A", "checksum": "0" * 64}],
    )

    with pytest.raises(ManifestError, match="checksum mismatch"):
        load_documents(tmp_path, load_manifest(manifest))


def test_matching_checksum_is_accepted(tmp_path: Path) -> None:
    import hashlib

    text = "# A\n\nreal content"
    (tmp_path / "a.md").write_text(text, encoding="utf-8")
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    manifest = _write(
        tmp_path / "manifest.json",
        [{"source_id": "a", "path": "a.md", "title": "A", "checksum": checksum}],
    )

    assert load_documents(tmp_path, load_manifest(manifest))[0].text == text


def test_missing_document_is_reported_clearly(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path / "manifest.json", [{"source_id": "a", "path": "a.md", "title": "A"}]
    )

    with pytest.raises(ManifestError, match="missing corpus document"):
        load_documents(tmp_path, load_manifest(manifest))

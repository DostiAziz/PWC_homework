from pathlib import Path

from customer_support.rag.ingest import CorpusChunk
from customer_support.rag.lexical import LexicalIndex


def make_chunk(chunk_id: str, source_id: str, text: str) -> CorpusChunk:
    return CorpusChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        title="Support guide",
        heading="Errors",
        original_text=text,
        context="Technical support error reference.",
        embedding_text=f"Technical support error reference. {text}",
        token_count=5,
        chunk_index=0,
        document_version="1",
        document_checksum="abc",
    )


def test_lexical_index_sync_search_and_delete(tmp_path: Path) -> None:
    index = LexicalIndex(tmp_path / "lexical.sqlite3")
    chunks = (
        make_chunk("exact", "guide", "Error TS-999 means the certificate expired."),
        make_chunk("general", "guide", "General support troubleshooting guidance."),
    )

    index.sync(chunks, {"guide"})
    assert index.search("TS-999")[0] == "exact"
    index.delete_source("guide")
    assert index.search("TS-999") == []

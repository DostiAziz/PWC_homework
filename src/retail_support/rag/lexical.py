from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from retail_support.rag.ingest import CorpusChunk


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

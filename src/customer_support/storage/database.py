from __future__ import annotations

import sqlite3
from pathlib import Path


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        schema = Path(__file__).with_name("retail_schema.sql").read_text(encoding="utf-8")
        with self.connect() as connection:
            connection.executescript(schema)

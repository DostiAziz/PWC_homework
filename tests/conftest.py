from pathlib import Path

import pytest

from scripts.seed_retail_data import seed
from storage.database import Database


@pytest.fixture
def retail_db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "retail.sqlite3")
    database.initialize()
    seed(database)
    return database

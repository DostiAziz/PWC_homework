from pathlib import Path

import pytest

from customer_support.storage.database import Database
from scripts.seed_retail_data import seed


@pytest.fixture
def retail_db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "retail.sqlite3")
    database.initialize()
    seed(database)
    return database

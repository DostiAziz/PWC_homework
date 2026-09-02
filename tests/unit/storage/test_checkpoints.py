from pathlib import Path

from pwc_support.storage.checkpoints import sqlite_checkpointer


def test_sqlite_checkpointer_creates_parent_directory(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "state" / "checkpoints.sqlite3"

    with sqlite_checkpointer(checkpoint_path) as saver:
        assert saver is not None

    assert checkpoint_path.exists()

from pathlib import Path

import pytest
from pydantic import ValidationError

from pwc_support.config import Settings


def test_defaults_match_local_mac_profile(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)

    assert settings.generation_model == "gpt-oss:20b"
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.num_ctx == 8192
    assert settings.max_parallel_generations == 1
    assert settings.operations_db == tmp_path / "state" / "operations.sqlite3"


def test_version_one_rejects_non_english() -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({"supported_language": "de"})

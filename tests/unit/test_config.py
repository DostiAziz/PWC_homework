from pathlib import Path

import pytest

from pwc_support.config import Settings


def test_defaults_point_to_new_retail_database(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)

    assert settings.retail_db == tmp_path / "state" / "retail-support-v1.sqlite3"
    assert settings.collection_name == "retail_support_v1_nomic_768_cosine"
    assert settings.generation_model == "gpt-oss:20b"
    assert settings.embedding_model == "nomic-embed-text"


def test_defaults_match_local_mac_profile(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)

    assert settings.generation_model == "gpt-oss:20b"
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.num_ctx == 8192
    assert settings.max_parallel_generations == 1
    assert settings.retail_db == tmp_path / "state" / "retail-support-v1.sqlite3"


def test_environment_wires_runtime_limits_and_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PWC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PWC_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("PWC_NUM_CTX", "4096")
    monkeypatch.setenv("PWC_SCHEMA_TOKENS", "128")
    monkeypatch.setenv("PWC_REQUEST_TIMEOUT_SECONDS", "7.5")
    monkeypatch.setenv("PWC_MAX_PARALLEL_GENERATIONS", "2")

    settings = Settings.from_env()

    assert settings.artifacts_dir == tmp_path / "artifacts"
    assert settings.num_ctx == 4096
    assert settings.schema_tokens == 128
    assert settings.request_timeout_seconds == 7.5
    assert settings.max_parallel_generations == 2

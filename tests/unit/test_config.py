from pathlib import Path

import pytest

from config import Settings


def test_defaults_point_to_new_retail_database(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)

    assert settings.retail_db == tmp_path / "state" / "retail-support-v1.sqlite3"
    assert settings.collection_name == "customer_support_v1_hf_384_cosine"
    assert settings.generation_model == "gpt-oss:20b"
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"


def test_defaults_match_local_mac_profile(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)

    assert settings.generation_model == "gpt-oss:20b"
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
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


def test_retail_environment_wires_runtime_limits_and_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RETAIL_DATA_DIR", str(tmp_path / "retail-data"))
    monkeypatch.setenv("RETAIL_ARTIFACTS_DIR", str(tmp_path / "retail-artifacts"))
    monkeypatch.setenv("RETAIL_NUM_CTX", "4096")
    monkeypatch.setenv("RETAIL_SCHEMA_TOKENS", "256")
    monkeypatch.setenv("RETAIL_REQUEST_TIMEOUT_SECONDS", "15.0")
    monkeypatch.setenv("RETAIL_MAX_PARALLEL_GENERATIONS", "2")

    settings = Settings.from_env()

    assert settings.data_dir == tmp_path / "retail-data"
    assert settings.artifacts_dir == tmp_path / "retail-artifacts"
    assert settings.num_ctx == 4096
    assert settings.schema_tokens == 256
    assert settings.request_timeout_seconds == 15.0
    assert settings.max_parallel_generations == 2


def test_customer_environment_wires_runtime_limits_and_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CUSTOMER_DATA_DIR", str(tmp_path / "customer-data"))
    monkeypatch.setenv("CUSTOMER_ARTIFACTS_DIR", str(tmp_path / "customer-artifacts"))
    monkeypatch.setenv("CUSTOMER_NUM_CTX", "2048")
    monkeypatch.setenv("CUSTOMER_SCHEMA_TOKENS", "512")
    monkeypatch.setenv("CUSTOMER_REQUEST_TIMEOUT_SECONDS", "30.0")
    monkeypatch.setenv("CUSTOMER_MAX_PARALLEL_GENERATIONS", "1")

    settings = Settings.from_env()

    assert settings.data_dir == tmp_path / "customer-data"
    assert settings.artifacts_dir == tmp_path / "customer-artifacts"
    assert settings.num_ctx == 2048
    assert settings.schema_tokens == 512
    assert settings.request_timeout_seconds == 30.0
    assert settings.max_parallel_generations == 1

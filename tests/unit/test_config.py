import json
from pathlib import Path

import pytest

from config import Settings


def test_defaults_point_to_new_retail_database(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)

    assert settings.retail_db == tmp_path / "state" / "retail-support-v1.sqlite3"
    assert settings.collection_name == "retail_support_v1_hf_384_cosine"
    assert settings.generation_model == "gpt-oss:20b"
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"


def test_defaults_match_local_mac_profile(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)

    assert settings.generation_model == "gpt-oss:20b"
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.max_parallel_generations == 1
    assert settings.retail_db == tmp_path / "state" / "retail-support-v1.sqlite3"


def test_environment_wires_runtime_limits_and_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PWC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PWC_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("PWC_ANSWER_TOKENS", "128")
    monkeypatch.setenv("PWC_REQUEST_TIMEOUT_SECONDS", "7.5")
    monkeypatch.setenv("PWC_MAX_PARALLEL_GENERATIONS", "2")

    settings = Settings.from_env()

    assert settings.artifacts_dir == tmp_path / "artifacts"
    assert settings.answer_tokens == 128
    assert settings.request_timeout_seconds == 7.5
    assert settings.max_parallel_generations == 2


def test_retail_environment_wires_runtime_limits_and_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RETAIL_DATA_DIR", str(tmp_path / "retail-data"))
    monkeypatch.setenv("RETAIL_ARTIFACTS_DIR", str(tmp_path / "retail-artifacts"))
    monkeypatch.setenv("RETAIL_ANSWER_TOKENS", "256")
    monkeypatch.setenv("RETAIL_REQUEST_TIMEOUT_SECONDS", "15.0")
    monkeypatch.setenv("RETAIL_MAX_PARALLEL_GENERATIONS", "2")

    settings = Settings.from_env()

    assert settings.data_dir == tmp_path / "retail-data"
    assert settings.artifacts_dir == tmp_path / "retail-artifacts"
    assert settings.answer_tokens == 256
    assert settings.request_timeout_seconds == 15.0
    assert settings.max_parallel_generations == 2


def test_customer_environment_wires_runtime_limits_and_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CUSTOMER_DATA_DIR", str(tmp_path / "customer-data"))
    monkeypatch.setenv("CUSTOMER_ARTIFACTS_DIR", str(tmp_path / "customer-artifacts"))
    monkeypatch.setenv("CUSTOMER_ANSWER_TOKENS", "512")
    monkeypatch.setenv("CUSTOMER_REQUEST_TIMEOUT_SECONDS", "30.0")
    monkeypatch.setenv("CUSTOMER_MAX_PARALLEL_GENERATIONS", "1")

    settings = Settings.from_env()

    assert settings.data_dir == tmp_path / "customer-data"
    assert settings.artifacts_dir == tmp_path / "customer-artifacts"
    assert settings.answer_tokens == 512
    assert settings.request_timeout_seconds == 30.0
    assert settings.max_parallel_generations == 1


def test_retrieval_config_revalidates_merged_constraints(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.json"
    path.write_text(
        json.dumps({"chunk_size_tokens": 100, "chunk_overlap_tokens": 100}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="chunk overlap must be smaller"):
        Settings().with_retrieval_config(path)


def test_retrieval_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.json"
    path.write_text(json.dumps({"ignored_typo": 3}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown retrieval configuration fields: ignored_typo"):
        Settings().with_retrieval_config(path)

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field, model_validator


class Settings(BaseModel):
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    ollama_base_url: str = "http://127.0.0.1:11434"
    generation_model: str = "gpt-oss:20b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_mode: Literal["persistent", "http"] = "persistent"
    chroma_host: str = "127.0.0.1"
    chroma_port: int = Field(default=8000, ge=1, le=65535)
    collection_name: str = "retail_support_v1_hf_384_cosine"
    num_ctx: int = Field(default=8192, ge=2048, le=32768)
    answer_tokens: int = Field(default=512, ge=64, le=1024)
    schema_tokens: int = Field(default=1024, ge=64, le=4096)
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    max_parallel_generations: int = Field(default=1, ge=1, le=2)
    embedding_batch_size: int = Field(default=16, ge=1, le=64)
    chunk_size_tokens: int = Field(default=300, ge=20, le=2000)
    chunk_overlap_tokens: int = Field(default=50, ge=0, le=500)
    context_document_max_chars: int = Field(default=24000, ge=2000, le=100000)
    top_k: int = Field(default=6, ge=1, le=20)
    minimum_similarity: float = Field(default=0.45, ge=0.0, le=1.0)
    max_selected_hits: int = Field(default=4, ge=1, le=10)
    max_evidence_chars: int = Field(default=6000, ge=500, le=40000)
    retail_db_path: Path | None = None

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def lexical_db(self) -> Path:
        return self.data_dir / "state" / "lexical.sqlite3"

    @property
    def retail_db(self) -> Path:
        return self.retail_db_path or self.data_dir / "state" / "retail-support-v1.sqlite3"

    def with_retrieval_config(self, path: Path) -> Settings:
        """Overlay the reviewable retrieval configuration file onto these settings."""
        if not path.exists():
            return self
        payload = json.loads(path.read_text(encoding="utf-8"))
        fields = set(type(self).model_fields)
        return self.model_copy(
            update={key: value for key, value in payload.items() if key in fields}
        )

    @model_validator(mode="after")
    def validate_constraints(self) -> Settings:
        if self.max_parallel_generations > 1 and self.num_ctx > 8192:
            raise ValueError("parallel generation above one requires num_ctx <= 8192")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk overlap must be smaller than chunk size")
        return self

    @classmethod
    def from_env(cls) -> Settings:
        def _get(key: str, default: str | None = None) -> str | None:
            return os.getenv(
                f"CUSTOMER_{key}",
                os.getenv(f"RETAIL_{key}", os.getenv(f"PWC_{key}", default)),
            )

        retail_db_override = _get("DB_PATH", os.getenv("PWC_RETAIL_DB_PATH"))
        return cls(
            data_dir=Path(_get("DATA_DIR", "data") or "data"),
            artifacts_dir=Path(_get("ARTIFACTS_DIR", "artifacts") or "artifacts"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            chroma_mode=cast(Literal["persistent", "http"], os.getenv("CHROMA_MODE", "persistent")),
            chroma_host=os.getenv("CHROMA_HOST", "127.0.0.1"),
            chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
            chunk_size_tokens=int(_get("CHUNK_SIZE_TOKENS", "300") or "300"),
            chunk_overlap_tokens=int(_get("CHUNK_OVERLAP_TOKENS", "50") or "50"),
            generation_model=_get("GENERATION_MODEL", "gpt-oss:20b") or "gpt-oss:20b",
            embedding_model=_get(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
            or "sentence-transformers/all-MiniLM-L6-v2",
            answer_tokens=int(_get("ANSWER_TOKENS", "512") or "512"),
            num_ctx=int(_get("NUM_CTX", "8192") or "8192"),
            schema_tokens=int(_get("SCHEMA_TOKENS", "1024") or "1024"),
            request_timeout_seconds=float(_get("REQUEST_TIMEOUT_SECONDS", "120") or "120"),
            max_parallel_generations=int(_get("MAX_PARALLEL_GENERATIONS", "1") or "1"),
            retail_db_path=Path(retail_db_override) if retail_db_override else None,
        )

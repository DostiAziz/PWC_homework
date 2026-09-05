from __future__ import annotations

import json
import os
from pathlib import Path
from typing import ClassVar, Literal, cast

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

from observability import configure_langsmith

load_dotenv()
configure_langsmith()


class Settings(BaseModel):
    RETRIEVAL_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "top_k",
            "minimum_similarity",
            "max_selected_hits",
            "max_evidence_chars",
            "chunk_size_tokens",
            "chunk_overlap_tokens",
            "embedding_batch_size",
            "context_document_max_chars",
        }
    )
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    ollama_base_url: str = "http://127.0.0.1:11434"
    generation_model: str = "gpt-oss:20b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_mode: Literal["persistent", "http"] = "persistent"
    chroma_host: str = "127.0.0.1"
    chroma_port: int = Field(default=8000, ge=1, le=65535)
    collection_name: str = "retail_support_v1_hf_384_cosine"
    answer_tokens: int = Field(default=512, ge=64, le=1024)
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    max_parallel_generations: int = Field(default=1, ge=1, le=2)
    embedding_batch_size: int = Field(default=16, ge=1, le=64)
    chunk_size_tokens: int = Field(default=300, ge=20, le=2000)
    chunk_overlap_tokens: int = Field(default=50, ge=0, le=500)
    context_document_max_chars: int = Field(default=24000, ge=2000, le=100000)
    top_k: int = Field(default=6, ge=1, le=20)
    minimum_similarity: float = Field(default=0.30, ge=0.0, le=1.0)
    max_selected_hits: int = Field(default=6, ge=1, le=10)
    max_evidence_chars: int = Field(default=6000, ge=500, le=40000)
    retail_db_path: Path | None = None
    langchain_tracing_v2: bool = False
    langchain_project: str = "retail-support"
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str | None = None

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
        if not isinstance(payload, dict):
            raise ValueError("Retrieval configuration must be a JSON object")
        unknown = sorted(set(payload) - self.RETRIEVAL_FIELDS)
        if unknown:
            raise ValueError(f"Unknown retrieval configuration fields: {', '.join(unknown)}")
        merged = self.model_dump() | payload
        return type(self).model_validate(merged)

    @model_validator(mode="after")
    def validate_constraints(self) -> Settings:
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
            request_timeout_seconds=float(_get("REQUEST_TIMEOUT_SECONDS", "120") or "120"),
            max_parallel_generations=int(_get("MAX_PARALLEL_GENERATIONS", "1") or "1"),
            retail_db_path=Path(retail_db_override) if retail_db_override else None,
            langchain_tracing_v2=(
                (
                    os.getenv("LANGCHAIN_TRACING_V2")
                    or os.getenv("LANGSMITH_TRACING")
                    or "false"
                ).lower()
                in ("true", "1", "yes")
            ),
            langchain_project=(
                os.getenv("LANGCHAIN_PROJECT")
                or os.getenv("LANGSMITH_PROJECT")
                or "retail-support"
            ),
            langchain_endpoint=(
                os.getenv("LANGCHAIN_ENDPOINT")
                or os.getenv("LANGSMITH_ENDPOINT")
                or "https://api.smith.langchain.com"
            ),
            langchain_api_key=(
                os.getenv("LANGCHAIN_API_KEY")
                or os.getenv("LANGSMITH_API_KEY")
            ),
        )

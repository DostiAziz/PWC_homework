from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field, model_validator


class Settings(BaseModel):
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    ollama_base_url: str = "http://127.0.0.1:11434"
    generation_model: str = "gpt-oss:20b"
    embedding_model: str = "nomic-embed-text"
    chroma_mode: Literal["persistent", "http"] = "persistent"
    chroma_host: str = "127.0.0.1"
    chroma_port: int = Field(default=8000, ge=1, le=65535)
    collection_name: str = "pwc_support_v2_nomic_768_cosine"
    num_ctx: int = Field(default=8192, ge=2048, le=32768)
    answer_tokens: int = Field(default=512, ge=64, le=1024)
    schema_tokens: int = Field(default=256, ge=64, le=512)
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    max_parallel_generations: int = Field(default=1, ge=1, le=2)
    embedding_batch_size: int = Field(default=16, ge=1, le=64)
    chunk_size_tokens: int = Field(default=300, ge=20, le=2000)
    chunk_overlap_tokens: int = Field(default=50, ge=0, le=500)
    context_document_max_chars: int = Field(default=24000, ge=2000, le=100000)
    max_planned_tasks: int = Field(default=4, ge=1, le=4)
    top_k: int = Field(default=6, ge=1, le=20)
    minimum_similarity: float = Field(default=0.45, ge=0.0, le=1.0)
    max_selected_hits: int = Field(default=4, ge=1, le=10)
    max_evidence_chars: int = Field(default=6000, ge=500, le=40000)
    retail_db_path: Path | None = None
    refund_auto_approval_limit: Decimal = Field(default=Decimal("100.00"), ge=0)
    return_window_days: int = Field(default=30, ge=1, le=365)
    reviewer_ids: tuple[str, ...] = ("specialist-1",)
    conversation_memory_max_chars: int = Field(default=12000, ge=500, le=100000)
    specialist_max_steps: int = Field(default=8, ge=1, le=50)

    @property
    def operations_db(self) -> Path:
        return self.data_dir / "state" / "operations.sqlite3"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def lexical_db(self) -> Path:
        return self.data_dir / "state" / "lexical.sqlite3"

    @property
    def mailbox_path(self) -> Path:
        return self.data_dir / "state" / "mailbox.json"

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
    def protect_mac_profile(self) -> Settings:
        if self.retail_db_path is None:
            object.__setattr__(self, "retail_db_path", self.data_dir / "state" / "retail.sqlite3")
        if self.max_parallel_generations > 1 and self.num_ctx > 8192:
            raise ValueError("parallel generation above one requires num_ctx <= 8192")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk overlap must be smaller than chunk size")
        return self

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            data_dir=Path(os.getenv("PWC_DATA_DIR", "data")),
            artifacts_dir=Path(os.getenv("PWC_ARTIFACTS_DIR", "artifacts")),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            chroma_mode=cast(Literal["persistent", "http"], os.getenv("CHROMA_MODE", "persistent")),
            chroma_host=os.getenv("CHROMA_HOST", "127.0.0.1"),
            chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
            chunk_size_tokens=int(os.getenv("PWC_CHUNK_SIZE_TOKENS", "300")),
            chunk_overlap_tokens=int(os.getenv("PWC_CHUNK_OVERLAP_TOKENS", "50")),
            generation_model=os.getenv("PWC_GENERATION_MODEL", "gpt-oss:20b"),
            embedding_model=os.getenv("PWC_EMBEDDING_MODEL", "nomic-embed-text"),
            answer_tokens=int(os.getenv("PWC_ANSWER_TOKENS", "512")),
            num_ctx=int(os.getenv("PWC_NUM_CTX", "8192")),
            schema_tokens=int(os.getenv("PWC_SCHEMA_TOKENS", "256")),
            request_timeout_seconds=float(os.getenv("PWC_REQUEST_TIMEOUT_SECONDS", "120")),
            max_parallel_generations=int(os.getenv("PWC_MAX_PARALLEL_GENERATIONS", "1")),
            reviewer_ids=tuple(
                item.strip()
                for item in os.getenv("PWC_REVIEWER_IDS", "specialist-1").split(",")
                if item.strip()
            ),
            conversation_memory_max_chars=int(
                os.getenv("PWC_CONVERSATION_MEMORY_MAX_CHARS", "12000")
            ),
            specialist_max_steps=int(os.getenv("PWC_SPECIALIST_MAX_STEPS", "8")),
        )

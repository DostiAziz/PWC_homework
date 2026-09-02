from __future__ import annotations

import os
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
    collection_name: str = "pwc_support_v1_nomic_768"
    supported_language: Literal["en"] = "en"
    num_ctx: int = Field(default=8192, ge=2048, le=32768)
    answer_tokens: int = Field(default=512, ge=64, le=1024)
    schema_tokens: int = Field(default=256, ge=64, le=512)
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    max_parallel_generations: int = Field(default=1, ge=1, le=2)
    embedding_batch_size: int = Field(default=16, ge=1, le=64)
    max_planned_tasks: int = Field(default=4, ge=1, le=4)
    max_revisions: int = Field(default=2, ge=0, le=2)

    @property
    def operations_db(self) -> Path:
        return self.data_dir / "state" / "operations.sqlite3"

    @property
    def checkpoints_db(self) -> Path:
        return self.data_dir / "state" / "checkpoints.sqlite3"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"

    @model_validator(mode="after")
    def protect_mac_profile(self) -> Settings:
        if self.max_parallel_generations > 1 and self.num_ctx > 8192:
            raise ValueError("parallel generation above one requires num_ctx <= 8192")
        return self

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            data_dir=Path(os.getenv("PWC_DATA_DIR", "data")),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            chroma_mode=cast(Literal["persistent", "http"], os.getenv("CHROMA_MODE", "persistent")),
            chroma_host=os.getenv("CHROMA_HOST", "127.0.0.1"),
            chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        )

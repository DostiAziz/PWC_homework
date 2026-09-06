from __future__ import annotations

from typing import Literal

from pydantic import Field

from domain.base import DomainModel
from domain.rag import Citation


class TimingSpan(DomainModel):
    name: str
    parent_id: str | None = None
    duration_ms: float = Field(ge=0)
    exclusive_ms: float = Field(ge=0)


class ChatReply(DomainModel):
    message: str
    citations: tuple[Citation, ...] = ()
    steps: tuple[str, ...] = ()
    awaiting_confirmation: bool = False
    preview: str = ""
    total_duration_ms: float = Field(ge=0, default=0.0)
    status: Literal[
        "answered",
        "awaiting_confirmation",
        "insufficient_evidence",
        "unavailable",
        "iteration_limit",
        "invalid_request",
    ] = "answered"
    spans: tuple[TimingSpan, ...] = ()
    request_id: str = ""

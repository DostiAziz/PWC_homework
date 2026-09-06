from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, HttpUrl, StringConstraints

from domain.base import DomainModel


class Citation(DomainModel):
    source_id: str
    chunk_id: str
    marker: str
    title: str
    canonical_url: HttpUrl | None = None
    heading: str
    excerpt: str
    similarity: float = Field(ge=-1.0, le=1.0)


class RetrievalHit(DomainModel):
    source_id: str
    chunk_id: str
    title: str
    text: str
    heading: str
    canonical_url: HttpUrl | None = None
    similarity: float = Field(ge=-1.0, le=1.0)
    language: str = "en"
    source_status: str = "active"
    source_type: str = "public_summary"


class RetrievalBatch(DomainModel):
    hits: tuple[RetrievalHit, ...] = ()
    used_filter_fallback: bool = False
    evidence_conflict: bool = False


class RagRequest(DomainModel):
    question: Annotated[str, StringConstraints(min_length=1, max_length=4000)]
    language: Literal["en"] = "en"


class RagResult(DomainModel):
    status: Literal["answered", "insufficient_evidence"]
    answer: str = ""
    citations: tuple[Citation, ...] = ()
    hits: tuple[RetrievalHit, ...] = ()
    used_filter_fallback: bool = False
    evidence_conflict: bool = False

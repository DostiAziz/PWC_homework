from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
)


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProductSummary(DomainModel):
    product_id: str
    name: str
    category: str
    price: Decimal = Field(ge=0)
    currency: str
    stock: int = Field(ge=0)


class OfferSummary(DomainModel):
    offer_id: str
    product_id: str
    name: str
    description: str
    list_price: Decimal = Field(ge=0)
    discount_percent: Decimal = Field(ge=0, le=100)
    effective_price: Decimal = Field(ge=0)
    currency: str


class OrderSummary(DomainModel):
    order_id: str
    customer_id: str
    status: str
    fulfilment_status: str
    total: Decimal
    currency: str
    version: int = Field(ge=1)


class CancellationPreview(DomainModel):
    confirmation_token: str = Field(min_length=8, max_length=64)
    order_id: str
    customer_id: str
    expected_version: int = Field(ge=1)
    expected_total: Decimal = Field(default=Decimal("79.99"), ge=0)
    expected_currency: str = Field(default="EUR")
    summary: str


class PendingCancellation(DomainModel):
    tool_call_id: str
    preview: CancellationPreview


class CancellationResult(DomainModel):
    order_id: str
    status: Literal["cancelled"]
    replayed: bool


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


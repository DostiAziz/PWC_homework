"""Validated domain contracts."""

from domain.base import DomainModel
from domain.chat import ChatReply, TimingSpan
from domain.rag import (
    Citation,
    RagRequest,
    RagResult,
    RetrievalBatch,
    RetrievalHit,
)
from domain.retail import (
    CancellationPreview,
    CancellationResult,
    OfferSummary,
    OrderSummary,
    PendingCancellation,
    ProductSummary,
)

__all__ = [
    "CancellationPreview",
    "CancellationResult",
    "ChatReply",
    "Citation",
    "DomainModel",
    "OfferSummary",
    "OrderSummary",
    "PendingCancellation",
    "ProductSummary",
    "RagRequest",
    "RagResult",
    "RetrievalBatch",
    "RetrievalHit",
    "TimingSpan",
]

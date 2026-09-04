"""Typed contracts shared by the classifier, specialist subgraphs, and supervisor.

This module is a thin re-export of `pwc_support.domain.models`, which is the single
source of truth for these types. Repositories under `pwc_support.storage` return these
same types, and storage must not import from `pwc_support.agents`, so the definitions
live in `domain.models` and are re-exported here for the agents package's own callers.
"""

from __future__ import annotations

from pwc_support.domain.models import (
    CancellationExecutionResult,
    CancellationReviewPacket,
    ConversationMemory,
    OfferSummary,
    OrderInvestigation,
    PolicyFinding,
    ProductRecommendation,
    RefundReviewPacket,
    ReturnInvestigation,
    ReturnReviewPacket,
    RoutingDecision,
    RoutingTask,
    SpecialistName,
    SpecialistResult,
    SpecialistStatus,
    SupportIntent,
)

__all__ = [
    "CancellationExecutionResult",
    "CancellationReviewPacket",
    "ConversationMemory",
    "OfferSummary",
    "OrderInvestigation",
    "PolicyFinding",
    "ProductRecommendation",
    "RefundReviewPacket",
    "ReturnInvestigation",
    "ReturnReviewPacket",
    "RoutingDecision",
    "RoutingTask",
    "SpecialistName",
    "SpecialistResult",
    "SpecialistStatus",
    "SupportIntent",
]

"""Typed contracts shared by the classifier, specialist subgraphs, and supervisor.

The domain-model types below are a thin re-export of `pwc_support.domain.models`, which is
the single source of truth for these types. Repositories under `pwc_support.storage` return
these same types, and storage must not import from `pwc_support.agents`, so the definitions
live in `domain.models` and are re-exported here for the agents package's own callers.

The classifier symbols (`IntentClassifier`, `OllamaIntentClassifier`,
`validate_routing_decision`) are defined in the sibling `agents.classifier` module, not
`domain.models` — they are re-exported here too so callers only need one import surface for
the agents package's public API.
"""

from __future__ import annotations

from pwc_support.agents.classifier import (
    IntentClassifier,
    OllamaIntentClassifier,
    validate_routing_decision,
)
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
    "IntentClassifier",
    "OfferSummary",
    "OllamaIntentClassifier",
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
    "validate_routing_decision",
]

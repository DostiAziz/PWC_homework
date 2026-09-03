from __future__ import annotations

import re
from dataclasses import dataclass

from pwc_support.domain.models import ReviewCategory, Route

GREETING = re.compile(
    r"^(hi|hello|hey|hiya|good (?:morning|afternoon|evening)|greetings)"
    r"(?: there| team| pwc| folks)?[\s!.,?]*$"
)
COURTESY = re.compile(r"^(thanks|thank you|cheers|ok|okay|got it|perfect|great)[\s!.,?]*$")


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    requires_review: bool
    categories: frozenset[ReviewCategory]


class ReviewPolicy:
    """Deterministic safety routing for cases requiring a human decision."""

    _patterns: tuple[tuple[ReviewCategory, tuple[str, ...]], ...] = (
        (
            ReviewCategory.CONFIDENTIALITY,
            (r"confidential", r"data breach", r"cyber ?security", r"expos(?:ed|ure)", r"leak"),
        ),
        (
            ReviewCategory.LEGAL_REGULATORY,
            (r"legal advice", r"regulat(?:or|ory)", r"compliance advice", r"lawsuit", r"sanction"),
        ),
        (
            ReviewCategory.COMPLAINT_ESCALATION,
            (r"complaint", r"escalat(?:e|ion)", r"speak to a person", r"manager", r"dissatisfied"),
        ),
        (
            ReviewCategory.EXTERNAL_ACTION,
            (r"send .* to", r"submit .* to", r"contact .* on our behalf", r"sign(?:ed)? report"),
        ),
        (
            ReviewCategory.PROFESSIONAL_JUDGEMENT,
            (r"audit conclusion", r"tax position", r"assurance opinion", r"professional advice"),
        ),
    )

    @classmethod
    def default(cls) -> ReviewPolicy:
        return cls()

    def evaluate(self, text: str) -> PolicyDecision:
        normalized = text.casefold()
        categories = frozenset(
            category
            for category, patterns in self._patterns
            if any(re.search(pattern, normalized) for pattern in patterns)
        )
        return PolicyDecision(requires_review=bool(categories), categories=categories)

    def classify_route(self, text: str) -> Route:
        """Deterministic first-pass routing before any model or retrieval call."""
        normalized = " ".join(text.split()).casefold()
        if not normalized:
            return Route.CLARIFY
        if self.evaluate(text).requires_review:
            return Route.REVIEW
        if GREETING.match(normalized) or COURTESY.match(normalized):
            return Route.GREETING
        if len(re.findall(r"[a-z0-9]+", normalized)) < 3:
            return Route.CLARIFY
        return Route.PLAN

    def after_retrieval(self, *, has_sufficient_evidence: bool) -> PolicyDecision:
        if has_sufficient_evidence:
            return PolicyDecision(requires_review=False, categories=frozenset())
        return PolicyDecision(
            requires_review=True,
            categories=frozenset({ReviewCategory.INSUFFICIENT_EVIDENCE}),
        )

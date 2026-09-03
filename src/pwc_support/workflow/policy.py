from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pwc_support.domain.models import (
    ReviewCategory,
    Route,
    SemanticRiskDecision,
    SemanticRiskRoute,
)

GREETING = re.compile(
    r"^(hi|hello|hey|hiya|good (?:morning|afternoon|evening)|greetings)"
    r"(?: there| team| pwc| folks)?[\s!.,?]*$"
)
COURTESY = re.compile(r"^(thanks|thank you|cheers|ok|okay|got it|perfect|great)[\s!.,?]*$")

PostRetrievalState = Literal[
    "sufficient",
    "insufficient",
    "conflicting",
    "citation_verification_failure",
]


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    requires_review: bool
    categories: frozenset[ReviewCategory]
    matched_rule_ids: tuple[str, ...]
    policy_version: str


@dataclass(frozen=True, slots=True)
class _PolicyRule:
    rule_id: str
    category: ReviewCategory
    pattern: re.Pattern[str]


def _rule(rule_id: str, category: ReviewCategory, pattern: str) -> _PolicyRule:
    return _PolicyRule(rule_id, category, re.compile(pattern))


class ReviewPolicy:
    """Versioned deterministic routing for cases requiring a human decision."""

    policy_version = "rules-v1"

    _rules: tuple[_PolicyRule, ...] = (
        _rule("confidentiality_confidential", ReviewCategory.CONFIDENTIALITY, r"confidential"),
        _rule("confidentiality_data_breach", ReviewCategory.CONFIDENTIALITY, r"data breach"),
        _rule("confidentiality_cybersecurity", ReviewCategory.CONFIDENTIALITY, r"cyber ?security"),
        _rule("confidentiality_exposure", ReviewCategory.CONFIDENTIALITY, r"expos(?:ed|ure)"),
        _rule("confidentiality_leak", ReviewCategory.CONFIDENTIALITY, r"leak"),
        _rule("legal_regulatory_legal_advice", ReviewCategory.LEGAL_REGULATORY, r"legal advice"),
        _rule("legal_regulatory_regulator", ReviewCategory.LEGAL_REGULATORY, r"regulat(?:or|ory)"),
        _rule(
            "legal_regulatory_compliance_advice",
            ReviewCategory.LEGAL_REGULATORY,
            r"compliance advice",
        ),
        _rule("legal_regulatory_lawsuit", ReviewCategory.LEGAL_REGULATORY, r"lawsuit"),
        _rule("legal_regulatory_sanction", ReviewCategory.LEGAL_REGULATORY, r"sanction"),
        _rule("complaint_escalation_complaint", ReviewCategory.COMPLAINT_ESCALATION, r"complaint"),
        _rule(
            "complaint_escalation_escalate",
            ReviewCategory.COMPLAINT_ESCALATION,
            r"escalat(?:e|ion)",
        ),
        _rule(
            "complaint_escalation_human_request",
            ReviewCategory.COMPLAINT_ESCALATION,
            r"speak to a person",
        ),
        _rule("complaint_escalation_manager", ReviewCategory.COMPLAINT_ESCALATION, r"manager"),
        _rule(
            "complaint_escalation_dissatisfied",
            ReviewCategory.COMPLAINT_ESCALATION,
            r"dissatisfied",
        ),
        _rule("external_action_send", ReviewCategory.EXTERNAL_ACTION, r"send .* to"),
        _rule("external_action_submit", ReviewCategory.EXTERNAL_ACTION, r"submit .* to"),
        _rule(
            "external_action_contact",
            ReviewCategory.EXTERNAL_ACTION,
            r"contact .* on our behalf",
        ),
        _rule(
            "external_action_signed_report",
            ReviewCategory.EXTERNAL_ACTION,
            r"sign(?:ed)? report",
        ),
        _rule(
            "external_action_confirmation",
            ReviewCategory.EXTERNAL_ACTION,
            r"\b(?:please proceed|go ahead|do that)\b",
        ),
        _rule(
            "professional_judgement_audit_conclusion",
            ReviewCategory.PROFESSIONAL_JUDGEMENT,
            r"audit conclusion",
        ),
        _rule(
            "professional_judgement_tax_position",
            ReviewCategory.PROFESSIONAL_JUDGEMENT,
            r"tax position",
        ),
        _rule(
            "professional_judgement_assurance_opinion",
            ReviewCategory.PROFESSIONAL_JUDGEMENT,
            r"assurance opinion",
        ),
        _rule(
            "professional_judgement_professional_advice",
            ReviewCategory.PROFESSIONAL_JUDGEMENT,
            r"professional advice",
        ),
    )

    @classmethod
    def default(cls) -> ReviewPolicy:
        return cls()

    def evaluate(self, text: str) -> PolicyDecision:
        normalized = text.casefold()
        matches = tuple(rule for rule in self._rules if rule.pattern.search(normalized))
        categories = frozenset(rule.category for rule in matches)
        return PolicyDecision(
            requires_review=bool(categories),
            categories=categories,
            matched_rule_ids=tuple(rule.rule_id for rule in matches),
            policy_version=self.policy_version,
        )

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

    def after_retrieval(
        self,
        *,
        evidence_state: PostRetrievalState | None = None,
        has_sufficient_evidence: bool | None = None,
    ) -> PolicyDecision:
        """Map evidence verification outcomes to deterministic review categories.

        ``has_sufficient_evidence`` remains accepted for the existing graph until its
        evidence gate migrates to the more precise ``evidence_state`` contract.
        """
        if evidence_state is None:
            if has_sufficient_evidence is None:
                raise ValueError("an evidence result is required")
            evidence_state = "sufficient" if has_sufficient_evidence else "insufficient"
        elif has_sufficient_evidence is not None:
            raise ValueError("provide evidence_state or has_sufficient_evidence, not both")

        category_by_state: dict[PostRetrievalState, ReviewCategory | None] = {
            "sufficient": None,
            "insufficient": ReviewCategory.INSUFFICIENT_EVIDENCE,
            "conflicting": ReviewCategory.CONFLICTING_EVIDENCE,
            "citation_verification_failure": ReviewCategory.CITATION_VERIFICATION_FAILURE,
        }
        category = category_by_state[evidence_state]
        if category is None:
            return PolicyDecision(False, frozenset(), (), self.policy_version)
        return PolicyDecision(
            requires_review=True,
            categories=frozenset({category}),
            matched_rule_ids=(f"post_retrieval_{evidence_state}",),
            policy_version=self.policy_version,
        )


def merge_routing(
    *,
    deterministic: PolicyDecision,
    semantic: SemanticRiskDecision | None,
) -> tuple[bool, frozenset[ReviewCategory]]:
    """Combine routing signals while preserving every deterministic escalation."""
    categories = set(deterministic.categories)
    if semantic is not None and semantic.route in {
        SemanticRiskRoute.REVIEW,
        SemanticRiskRoute.UNCERTAIN,
    }:
        categories.update(semantic.categories)
    return bool(categories), frozenset(categories)

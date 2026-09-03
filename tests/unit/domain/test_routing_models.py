import pytest
from pydantic import ValidationError

from pwc_support.domain.models import (
    MessageKind,
    OutboundMessage,
    ReviewCategory,
    SemanticRiskDecision,
    SemanticRiskRoute,
)


def test_routine_semantic_result_has_no_categories() -> None:
    result = SemanticRiskDecision(route=SemanticRiskRoute.ROUTINE)
    assert result.categories == frozenset()


def test_review_and_uncertain_require_known_categories() -> None:
    with pytest.raises(ValidationError):
        SemanticRiskDecision(route=SemanticRiskRoute.REVIEW)
    with pytest.raises(ValidationError):
        SemanticRiskDecision(
            route=SemanticRiskRoute.REVIEW,
            categories=frozenset({"not-a-category"}),
        )
    result = SemanticRiskDecision(
        route=SemanticRiskRoute.UNCERTAIN,
        categories=frozenset({ReviewCategory.OTHER_SENSITIVE_RISK}),
    )
    assert result.route is SemanticRiskRoute.UNCERTAIN


def test_reviewed_response_requires_case_and_version() -> None:
    with pytest.raises(ValidationError):
        OutboundMessage(
            delivery_key="d-1",
            kind=MessageKind.REVIEWED_RESPONSE,
            recipient="client@example.test",
            thread_id="thread-1",
            subject="Response",
            body="Approved response",
            payload_hash="hash",
        )

    message = OutboundMessage(
        delivery_key="d-1",
        kind=MessageKind.REVIEWED_RESPONSE,
        case_id="CASE-1",
        response_version=1,
        recipient="client@example.test",
        thread_id="thread-1",
        subject="Response",
        body="Approved response",
        payload_hash="hash",
    )
    assert message.response_version == 1

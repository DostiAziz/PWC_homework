from pwc_support.domain.models import ReviewCategory
from pwc_support.workflow.policy import ReviewPolicy


def test_sensitive_categories_are_routed_to_review() -> None:
    policy = ReviewPolicy.default()

    decision = policy.evaluate(
        "We may have exposed a confidential client document. Please investigate."
    )

    assert decision.requires_review is True
    assert ReviewCategory.CONFIDENTIALITY in decision.categories


def test_general_public_service_question_can_be_answered() -> None:
    decision = ReviewPolicy.default().evaluate(
        "What services does PwC provide to banks?"
    )

    assert decision.requires_review is False
    assert decision.categories == frozenset()


def test_policy_identifies_professional_judgement_and_external_action() -> None:
    decision = ReviewPolicy.default().evaluate(
        "Can you confirm our audit conclusion and send the signed report to the regulator?"
    )

    assert decision.requires_review is True
    assert ReviewCategory.PROFESSIONAL_JUDGEMENT in decision.categories
    assert ReviewCategory.EXTERNAL_ACTION in decision.categories


def test_empty_evidence_requires_review() -> None:
    decision = ReviewPolicy.default().after_retrieval(has_sufficient_evidence=False)

    assert decision.requires_review is True
    assert ReviewCategory.INSUFFICIENT_EVIDENCE in decision.categories

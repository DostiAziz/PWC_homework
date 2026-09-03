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
    decision = ReviewPolicy.default().evaluate("What services does PwC provide to banks?")

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


def test_greetings_and_courtesies_are_not_escalated() -> None:
    policy = ReviewPolicy.default()

    assert policy.classify_route("hi").value == "greeting"
    assert policy.classify_route("Good morning there!").value == "greeting"
    assert policy.classify_route("thanks").value == "greeting"


def test_short_input_is_clarified_and_full_questions_are_planned() -> None:
    policy = ReviewPolicy.default()

    assert policy.classify_route("insurance?").value == "clarify"
    assert policy.classify_route("").value == "clarify"
    assert policy.classify_route("What services does PwC provide to banks?").value == "plan"


def test_risk_language_still_wins_over_a_polite_opening() -> None:
    assert ReviewPolicy.default().classify_route("Hello, we had a data breach.").value == "review"


def test_action_confirmation_is_a_hard_external_action_match() -> None:
    decision = ReviewPolicy.default().evaluate("Please proceed with that submission.")

    assert ReviewCategory.EXTERNAL_ACTION in decision.categories
    assert "external_action_confirmation" in decision.matched_rule_ids
    assert decision.policy_version == "rules-v1"


def test_other_short_follow_up_still_requires_clarification() -> None:
    assert ReviewPolicy.default().classify_route("Which one?").value == "clarify"


def test_policy_records_stable_rule_ids_for_every_matching_category() -> None:
    decision = ReviewPolicy.default().evaluate("We had a confidential leak and need legal advice.")

    assert decision.matched_rule_ids == (
        "confidentiality_confidential",
        "confidentiality_leak",
        "legal_regulatory_legal_advice",
    )


def test_post_retrieval_failures_map_to_distinct_review_categories() -> None:
    policy = ReviewPolicy.default()

    insufficient = policy.after_retrieval(evidence_state="insufficient")
    conflicting = policy.after_retrieval(evidence_state="conflicting")
    citation_failure = policy.after_retrieval(evidence_state="citation_verification_failure")

    assert insufficient.categories == frozenset({ReviewCategory.INSUFFICIENT_EVIDENCE})
    assert conflicting.categories == frozenset({ReviewCategory.CONFLICTING_EVIDENCE})
    assert citation_failure.categories == frozenset({ReviewCategory.CITATION_VERIFICATION_FAILURE})

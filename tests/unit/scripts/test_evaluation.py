from pwc_support.domain.models import ChatReply, Citation
from scripts.run_evaluation import score_case


def test_tool_routing_scores_correctly() -> None:
    case = {
        "id": "order-status",
        "expected_tools": ["get_order_status"],
        "required_terms": ["shipped"],
        "required_sources": [],
        "forbidden_terms": [],
    }
    reply = ChatReply(
        message="Order ORD-5001 is shipped.",
        steps=("get_order_status",),
        total_duration_ms=10.0,
    )
    score = score_case(case, reply)
    assert score.checks["routing"] is True
    assert score.checks["terms"] is True


def test_cancel_safety_requires_confirmation_gate() -> None:
    case = {
        "id": "cancel-safe",
        "expected_tools": ["cancel_order"],
        "expect_confirmation": True,
        "required_terms": [],
        "required_sources": [],
        "forbidden_terms": [],
    }
    reply = ChatReply(
        message="Cancel order ORD-2001 for 79.99 EUR? Please answer yes or no.",
        awaiting_confirmation=True,
        steps=("cancel_order",),
        total_duration_ms=10.0,
    )
    score = score_case(case, reply)
    assert score.checks["safety"] is True


def test_score_case_rejects_unmapped_citation_marker() -> None:
    case = {
        "id": "bad-marker",
        "expected_tools": [],
        "required_sources": [],
        "required_terms": [],
        "forbidden_terms": [],
        "expect_citations": True,
        "expect_confirmation": False,
    }
    reply = ChatReply(
        message="Unsupported marker. [S9]",
        total_duration_ms=1.0,
    )

    assert not score_case(case, reply).checks["attribution"]


def test_score_case_valid_citation() -> None:
    citation = Citation(
        source_id="shipping-and-orders",
        chunk_id="shipping-1",
        marker="[S1]",
        title="Shipping and order support",
        heading="Delivery times",
        excerpt="Three to five business days.",
        similarity=0.9,
    )
    case = {
        "id": "shipping-time",
        "expected_tools": ["search_policies"],
        "required_sources": ["shipping-and-orders"],
        "required_terms": ["three"],
        "forbidden_terms": [],
        "expect_citations": True,
        "expect_confirmation": False,
    }
    reply = ChatReply(
        message="Shipping takes three to five business days. [S1]",
        citations=(citation,),
        steps=("search_policies",),
        total_duration_ms=12.0,
    )
    score = score_case(case, reply)
    assert score.passed
    assert all(score.checks.values())

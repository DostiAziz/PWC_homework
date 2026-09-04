from pwc_support.domain.models import ChatReply, Citation, Task, TaskKind
from scripts.run_evaluation import score_case


def test_score_case_checks_routing_sources_terms_and_pending_state() -> None:
    case = {
        "id": "shipping-time",
        "turns": ["How long is shipping?"],
        "expected_task_kinds": ["knowledge"],
        "required_sources": ["shipping-and-orders"],
        "required_terms": ["three"],
        "forbidden_terms": ["overnight"],
        "expect_citations": True,
        "expect_pending_cancellation": False,
    }
    citation = Citation(
        source_id="shipping-and-orders",
        chunk_id="shipping-1",
        marker="[S1]",
        title="Shipping and order support",
        heading="Delivery times",
        excerpt="Three to five business days.",
        similarity=0.9,
    )
    reply = ChatReply(
        message="Shipping takes three to five business days. [S1]",
        citations=(citation,),
        tasks=(Task(task_id="task-1", kind=TaskKind.KNOWLEDGE, request="shipping"),),
        total_duration_ms=12.0,
    )

    score = score_case(case, reply)

    assert score.passed
    assert all(score.checks.values())


def test_score_case_rejects_unmapped_citation_marker() -> None:
    case = {
        "id": "bad-marker",
        "turns": ["question"],
        "expected_task_kinds": ["knowledge"],
        "required_sources": [],
        "required_terms": [],
        "forbidden_terms": [],
        "expect_citations": True,
        "expect_pending_cancellation": False,
    }
    reply = ChatReply(
        message="Unsupported marker. [S9]",
        tasks=(Task(task_id="task-1", kind=TaskKind.KNOWLEDGE, request="question"),),
        total_duration_ms=1.0,
    )

    assert not score_case(case, reply).checks["attribution"]

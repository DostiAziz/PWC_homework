from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pwc_support.domain.models import (
    CaseRequest,
    Citation,
    OperationalEvent,
    ReviewCategory,
    ReviewDecisionKind,
    ReviewRequest,
)
from pwc_support.storage.database import Database
from pwc_support.storage.repositories import CaseRepository, ReviewRepository


def test_database_initializes_and_case_creation_is_idempotent(tmp_path: Path) -> None:
    database = Database(tmp_path / "operations.sqlite3")
    database.initialize()
    cases = CaseRepository(database)
    request = CaseRequest(
        conversation_id=uuid4(),
        client_id="client-1",
        category="confidentiality",
        summary="Potential document exposure",
    )

    first = cases.create(request, case_id="DEMO-001")
    second = cases.create(request, case_id="DEMO-001")

    assert first == second
    assert cases.get("DEMO-001").case_id == "DEMO-001"


def test_review_repository_lists_pending_and_records_decision(tmp_path: Path) -> None:
    database = Database(tmp_path / "operations.sqlite3")
    database.initialize()
    reviews = ReviewRepository(database)
    review = ReviewRequest(
        review_id=uuid4(),
        case_id="DEMO-002",
        run_id=uuid4(),
        categories=frozenset({ReviewCategory.LEGAL_REGULATORY}),
        original_message="Please confirm our regulatory position.",
        response_version=1,
    )

    reviews.create(review)
    assert reviews.list_pending()[0].review_id == review.review_id
    reviews.decide(
        review_id=review.review_id,
        decision_id=uuid4(),
        expected_version=review.response_version,
        reviewer_id="specialist-1",
        kind=ReviewDecisionKind.REJECT,
    )
    assert reviews.list_pending() == []


def test_events_are_stored_without_message_content(tmp_path: Path) -> None:
    database = Database(tmp_path / "operations.sqlite3")
    database.initialize()
    event = OperationalEvent(
        run_id=uuid4(),
        conversation_id=uuid4(),
        node="triage",
        event_type="completed",
        duration_ms=12.5,
        details={"route": "plan"},
        occurred_at=datetime.now(UTC),
    )

    database.record_event(event)

    assert database.list_events(event.run_id)[0].details == {"route": "plan"}


def test_review_survives_a_restart_with_evidence_and_delivery_metadata(tmp_path: Path) -> None:
    """A queued review remains actionable in a process that did not create it."""
    path = tmp_path / "operations.sqlite3"
    review = ReviewRequest(
        review_id=uuid4(),
        case_id="DEMO-003",
        run_id=uuid4(),
        categories=frozenset({ReviewCategory.CONFIDENTIALITY}),
        original_message="A confidential report may have leaked.",
        evidence=(
            Citation(
                source_id="pwc-global-services",
                chunk_id="chunk-1",
                marker="[S1]",
                title="PwC global services",
                heading="Services",
                excerpt="PwC describes assurance, tax and advisory services.",
                similarity=0.82,
            ),
        ),
        delivery_recipient="client@example.test",
        delivery_thread_id="thread-abc123",
        delivery_subject="Confidential report",
        response_version=1,
    )

    first = Database(path)
    first.initialize()
    ReviewRepository(first).create(review)

    # A separate Database instance stands in for the next process.
    restarted = Database(path)
    restarted.initialize()
    pending = ReviewRepository(restarted).list_pending()[0]

    assert [citation.source_id for citation in pending.evidence] == ["pwc-global-services"]
    assert pending.delivery_recipient == "client@example.test"
    assert pending.delivery_thread_id == "thread-abc123"
    assert pending.delivery_subject == "Confidential report"


def test_initialize_backfills_columns_on_a_database_from_an_earlier_version(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.sqlite3"
    legacy = Database(path)
    with legacy.connect() as connection:
        connection.execute(
            """
            CREATE TABLE review_requests (
                review_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                categories_json TEXT NOT NULL,
                original_message TEXT NOT NULL,
                proposed_reply_json TEXT,
                proposed_actions_json TEXT NOT NULL,
                response_version INTEGER NOT NULL,
                status TEXT NOT NULL
            )
            """
        )

    legacy.initialize()

    with legacy.connect() as connection:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(review_requests)").fetchall()
        }
    assert {"evidence_json", "checkpoint_id"} <= columns

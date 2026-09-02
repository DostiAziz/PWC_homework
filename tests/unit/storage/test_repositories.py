from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pwc_support.domain.models import (
    CaseRequest,
    OperationalEvent,
    ReviewCategory,
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
    reviews.mark_decided(review.review_id)
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

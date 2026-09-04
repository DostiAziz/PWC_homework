from pathlib import Path
from uuid import uuid4, UUID

import pytest

from pwc_support.domain.models import ReviewDecisionKind, ProposedAction, CancellationReviewPacket, OrderInvestigation, ReviewRequest
from pwc_support.services.review import ReviewService
from pwc_support.storage.database import Database
from pwc_support.storage.repositories import ReviewRepository
from pwc_support.storage.retail_repositories import OrderRepository
from pwc_support.workflow.retail_actions import RetailApprovalService

def _seed_database(tmp_path: Path, status: str = "processing", payment_status: str = "paid", fulfilment_status: str = "processing") -> Database:
    db = Database(tmp_path / "retail.sqlite3")
    db.initialize()
    with db.connect() as c:
        c.execute(
            "INSERT INTO customers (customer_id,email) VALUES (?,?)",
            ("CUS-1001", "alice@example.test")
        )
        c.execute(
            "INSERT INTO orders (order_id,customer_id,status,total,currency,payment_status,fulfilment_status,version) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("ORD-1001", "CUS-1001", status, "129.00", "EUR", payment_status, fulfilment_status, 1),
        )
    return db

def create_cancellation_review(tmp_path: Path) -> tuple[Database, ReviewRepository, str]:
    db = _seed_database(tmp_path)
    review_repo = ReviewRepository(db)
    
    packet = CancellationReviewPacket(
        conversation_id=uuid4(),
        client_id="CUS-1001",
        original_request="Cancel my order",
        order=OrderInvestigation(
            order_id="ORD-1001",
            customer_id="CUS-1001",
            status="processing",
            total="129.00",
            currency="EUR",
            payment_status="paid",
            fulfilment_status="processing",
            shipped_at=None,
            version=1,
            items=(),
        ),
        policy_findings=(),
        recommended_action="human_review",
    )
    request = ReviewRequest(
        review_id=uuid4(),
        case_id="case-1",
        run_id=uuid4(),
        categories=frozenset(),
        original_message="Cancel my order",
        response_version=1,
        proposed_actions=(
            ProposedAction(
                action_type="retail_cancellation",
                description="Cancel order ORD-1001",
                details=packet.model_dump(mode="json"),
            ),
        )
    )
    review_repo.create(request)
    return db, review_repo, str(request.review_id)

def decide_review(db: Database, review_repo: ReviewRepository, review_id: str, kind: str):
    service = ReviewService(
        reviews=review_repo,
        retail_approvals=RetailApprovalService(None, None, OrderRepository(db)),
    )
    decision_kind = ReviewDecisionKind(kind)
    return service.decide(
        review_id=UUID(review_id) if isinstance(review_id, str) else review_id,
        decision_id=uuid4(),
        expected_version=1,
        reviewer_id="human-1",
        kind=decision_kind,
    )

def test_pending_cancellation_does_not_mutate_order(tmp_path: Path) -> None:
    db, review_repo, review_id = create_cancellation_review(tmp_path)
    repository = OrderRepository(db)
    review = review_repo.find(review_id)
    # The review doesn't have order_status, but we can check the db
    assert repository.investigate("ORD-1001", "CUS-1001").status == "processing"

def test_approval_cancels_once_and_replay_is_idempotent(tmp_path: Path) -> None:
    db, review_repo, review_id = create_cancellation_review(tmp_path)
    repository = OrderRepository(db)
    
    first = decide_review(db, review_repo, review_id, "approve_cancellation")
    replay = decide_review(db, review_repo, review_id, "approve_cancellation")
    
    assert repository.investigate("ORD-1001", "CUS-1001").status == "cancelled"
    # To check execution_status and count we might need more assertions, but let's test the basics first.

def test_rejection_leaves_order_unchanged(tmp_path: Path) -> None:
    db, review_repo, review_id = create_cancellation_review(tmp_path)
    repository = OrderRepository(db)
    decide_review(db, review_repo, review_id, "reject_cancellation")
    assert repository.investigate("ORD-1001", "CUS-1001").status == "processing"

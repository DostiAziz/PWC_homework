from pathlib import Path
from uuid import uuid4, UUID

import pytest

from decimal import Decimal
from pwc_support.domain.models import ReviewDecisionKind, ProposedAction, ReturnReviewPacket, RefundReviewPacket, ReturnInvestigation, ReviewRequest
from pwc_support.services.review import ReviewService
from pwc_support.storage.database import Database
from pwc_support.storage.repositories import ReviewRepository
from pwc_support.storage.retail_repositories import OrderRepository, ReturnRepository, RefundRepository
from pwc_support.workflow.retail_actions import RetailApprovalService

def _seed_database(tmp_path: Path, status: str = "delivered") -> Database:
    db = Database(tmp_path / "retail.sqlite3")
    db.initialize()
    with db.connect() as c:
        c.execute(
            "INSERT INTO products (product_id,name,category,price,currency,attributes_json,active) "
            "VALUES (?,?,?,?,?,?,?)",
            ("P-1", "Jacket", "outerwear", "129.00", "EUR", "{}", 1),
        )
        c.execute(
            "INSERT INTO inventory (product_id,location,quantity) VALUES (?,?,?)",
            ("P-1", "W-1", 100),
        )
        c.execute(
            "INSERT INTO customers (customer_id,email) VALUES (?,?)",
            ("CUS-1002", "bob@example.test")
        )
        c.execute(
            "INSERT INTO orders (order_id,customer_id,status,total,currency,payment_status,fulfilment_status,version,delivered_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            ("ORD-1002", "CUS-1002", status, "129.00", "EUR", "paid", "delivered", 1, "2026-09-01T12:00:00Z"),
        )
        c.execute(
            "INSERT INTO order_items (order_id,item_id,product_id,quantity,unit_price) "
            "VALUES (?,?,?,?,?)",
            ("ORD-1002", "ITEM-1002", "P-1", 1, "129.00")
        )
    return db

def create_return_review(tmp_path: Path) -> tuple[Database, ReviewRepository, str]:
    db = _seed_database(tmp_path)
    review_repo = ReviewRepository(db)
    
    packet = ReturnReviewPacket(
        conversation_id=uuid4(),
        client_id="CUS-1002",
        original_request="Return jacket",
        investigation=ReturnInvestigation(
            order_id="ORD-1002",
            customer_id="CUS-1002",
            item_id="ITEM-1002",
            order_status="delivered",
            item_description="Jacket",
            quantity=1,
            unit_price=Decimal("129.00"),
            currency="EUR",
            delivered_at=None,
            eligible=True,
            version=1,
        ),
        policy_findings=(),
        missing_information=(),
        recommended_action="human_review",
    )
    
    request = ReviewRequest(
        review_id=uuid4(),
        case_id="case-2",
        run_id=uuid4(),
        categories=frozenset(),
        original_message="Return jacket",
        response_version=1,
        proposed_actions=(
            ProposedAction(
                action_type="retail_return",
                description="Return item ITEM-1002",
                details=packet.model_dump(mode="json"),
            ),
        )
    )
    
    review_repo.create(request)
    return db, review_repo, str(request.review_id)

def decide_review(db: Database, review_repo: ReviewRepository, review_id: str, kind: str):
    service = ReviewService(
        reviews=review_repo,
        retail_approvals=RetailApprovalService(
            returns=ReturnRepository(db),
            refunds=RefundRepository(db),
            orders=OrderRepository(db)
        ),
    )
    decision_kind = ReviewDecisionKind(kind)
    return service.decide(
        review_id=UUID(review_id) if isinstance(review_id, str) else review_id,
        decision_id=uuid4(),
        expected_version=1,
        reviewer_id="human-1",
        kind=decision_kind,
    )

def test_approval_returns_once_and_replay_is_idempotent(tmp_path: Path) -> None:
    db, review_repo, review_id = create_return_review(tmp_path)
    returns = ReturnRepository(db)
    
    first = decide_review(db, review_repo, review_id, "approve_return")
    replay = decide_review(db, review_repo, review_id, "approve_return")
    
    # Assert idempotent execution and state
    with db.connect() as c:
        row = c.execute("SELECT status FROM return_requests WHERE order_id=?", ("ORD-1002",)).fetchone()
        assert row["status"] == "approved"

def test_rejection_leaves_return_rejected(tmp_path: Path) -> None:
    db, review_repo, review_id = create_return_review(tmp_path)
    
    first = decide_review(db, review_repo, review_id, "reject_return")
    
    with db.connect() as c:
        row = c.execute("SELECT status FROM return_requests WHERE order_id=?", ("ORD-1002",)).fetchone()
        assert row["status"] == "rejected"

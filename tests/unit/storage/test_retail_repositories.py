from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from pwc_support.storage.database import Database
from pwc_support.storage.retail_repositories import (
    ProductRepository,
    RefundRepository,
    ReturnRepository,
)
from pwc_support.workflow.retail_actions import request_return


def _db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "retail.sqlite3")
    db.initialize()
    now = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    with db.connect() as c:
        c.execute(
            "INSERT INTO products VALUES (?,?,?,?,?,?,?)",
            ("P-1", "Jacket", "outerwear", "120", "EUR", "{}", 1),
        )
        c.execute("INSERT INTO inventory VALUES (?,?,?)", ("P-1", "WH-1", 3))
        c.execute("INSERT INTO customers VALUES (?,?)", ("C-1", "c@example.test"))
        c.execute(
            "INSERT INTO orders VALUES (?,?,?,?,?,?)",
            ("O-1", "C-1", "delivered", "120", "EUR", now),
        )
        c.execute("INSERT INTO order_items VALUES (?,?,?,?,?)", ("I-1", "O-1", "P-1", 1, "120"))
    return db


def test_inventory_and_offer_are_scoped_repository_operations(tmp_path: Path) -> None:
    db = _db(tmp_path)
    repo = ProductRepository(db)
    assert repo.inventory("P-1") == 3
    assert repo.inventory("P-1", "missing") == 0
    assert repo.offer("P-1") is None


def test_refund_proposal_requires_expected_version_for_status_change(tmp_path: Path) -> None:
    db = _db(tmp_path)
    returns = ReturnRepository(db)
    request, eligibility = request_return(
        returns, order_id="O-1", item_id="I-1", reason="size", idempotency_key="k"
    )
    assert request is not None and eligibility.eligible
    refunds = RefundRepository(db)
    proposal = refunds.propose(request.return_id)
    assert proposal["status"] == "pending_review"
    updated = refunds.transition(str(proposal["refund_id"]), "approved", expected_version=1)
    assert updated["status"] == "approved"
    with pytest.raises(ValueError, match="version"):
        refunds.transition(str(proposal["refund_id"]), "rejected", expected_version=1)

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from pwc_support.storage.database import Database
from pwc_support.storage.retail_repositories import (
    OrderRepository,
    ProductRepository,
    RefundRepository,
    ReturnRepository,
)
from pwc_support.workflow.retail_actions import request_return


def _db(tmp_path: Path) -> Database:
    """A single delivered order/item fixture, used by the return/refund tests below."""
    db = Database(tmp_path / "retail.sqlite3")
    db.initialize()
    now = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    with db.connect() as c:
        c.execute(
            "INSERT INTO products (product_id,name,category,price,currency,attributes_json,active) "
            "VALUES (?,?,?,?,?,?,?)",
            ("P-1", "Jacket", "outerwear", "120", "EUR", "{}", 1),
        )
        c.execute(
            "INSERT INTO inventory (product_id,location,quantity) VALUES (?,?,?)",
            ("P-1", "WH-1", 3),
        )
        c.execute(
            "INSERT INTO customers (customer_id,email) VALUES (?,?)", ("C-1", "c@example.test")
        )
        c.execute(
            "INSERT INTO orders ("
            "order_id,customer_id,status,total,currency,delivered_at,"
            "payment_status,fulfilment_status,shipped_at,cancelled_at,version"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("O-1", "C-1", "delivered", "120", "EUR", now, "paid", "delivered", now, None, 1),
        )
        c.execute(
            "INSERT INTO order_items (item_id,order_id,product_id,quantity,unit_price) "
            "VALUES (?,?,?,?,?)",
            ("I-1", "O-1", "P-1", 1, "120"),
        )
    return db


def _seed_database(
    tmp_path: Path,
    *,
    status: str = "processing",
    payment_status: str = "unpaid",
    fulfilment_status: str = "processing",
) -> Database:
    """A single ORD-1/CUS-1 order used by the investigation and cancellation tests."""
    db = Database(tmp_path / "retail.sqlite3")
    db.initialize()
    with db.connect() as c:
        c.execute(
            "INSERT INTO products (product_id,name,category,price,currency,attributes_json,active) "
            "VALUES (?,?,?,?,?,?,?)",
            ("P-1", "Jacket", "outerwear", "120", "EUR", "{}", 1),
        )
        c.execute(
            "INSERT INTO inventory (product_id,location,quantity) VALUES (?,?,?)",
            ("P-1", "WH-1", 3),
        )
        c.execute(
            "INSERT INTO customers (customer_id,email) VALUES (?,?)", ("CUS-1", "c@example.test")
        )
        c.execute(
            "INSERT INTO orders ("
            "order_id,customer_id,status,total,currency,delivered_at,"
            "payment_status,fulfilment_status,shipped_at,cancelled_at,version"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                "ORD-1",
                "CUS-1",
                status,
                "120",
                "EUR",
                None,
                payment_status,
                fulfilment_status,
                None,
                None,
                1,
            ),
        )
        c.execute(
            "INSERT INTO order_items (item_id,order_id,product_id,quantity,unit_price) "
            "VALUES (?,?,?,?,?)",
            ("I-1", "ORD-1", "P-1", 1, "120"),
        )
    return db


def _catalogue_database(tmp_path: Path) -> Database:
    """Two categories, varied stock, and two active offers on distinct products."""
    db = Database(tmp_path / "retail.sqlite3")
    db.initialize()
    with db.connect() as c:
        c.executemany(
            "INSERT INTO products (product_id,name,category,price,currency,attributes_json,active) "
            "VALUES (?,?,?,?,?,?,?)",
            [
                (
                    "P-OFFER-1",
                    "Rain Jacket",
                    "outerwear",
                    "100.00",
                    "EUR",
                    json.dumps({"waterproof": True}),
                    1,
                ),
                (
                    "P-OFFER-2",
                    "Trail Backpack",
                    "bags",
                    "80.00",
                    "EUR",
                    json.dumps({"capacity_l": 30}),
                    1,
                ),
                (
                    "P-PLAIN-1",
                    "Winter Jacket",
                    "outerwear",
                    "150.00",
                    "EUR",
                    json.dumps({"insulated": True}),
                    1,
                ),
                (
                    "P-OOS-1",
                    "Summer Jacket",
                    "outerwear",
                    "60.00",
                    "EUR",
                    json.dumps({"lightweight": True}),
                    1,
                ),
            ],
        )
        c.executemany(
            "INSERT INTO inventory (product_id,location,quantity) VALUES (?,?,?)",
            [
                ("P-OFFER-1", "WH-1", 5),
                ("P-OFFER-2", "WH-1", 8),
                ("P-PLAIN-1", "WH-1", 2),
                ("P-OOS-1", "WH-1", 0),
            ],
        )
        c.executemany(
            "INSERT INTO offers (offer_id,product_id,description,discount_percent,active) "
            "VALUES (?,?,?,?,?)",
            [
                ("OFFER-A", "P-OFFER-1", "10% off", "10", 1),
                ("OFFER-B", "P-OFFER-2", "20% off", "20", 1),
                ("OFFER-OLD", "P-PLAIN-1", "5% off expired", "5", 0),
            ],
        )
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


def test_investigation_returns_payment_and_fulfilment_facts(tmp_path: Path) -> None:
    database = _seed_database(
        tmp_path, status="processing", payment_status="paid", fulfilment_status="processing"
    )
    investigation = OrderRepository(database).investigate("ORD-1", "CUS-1")
    assert investigation is not None
    assert investigation.payment_status == "paid"
    assert investigation.fulfilment_status == "processing"
    assert investigation.version == 1


def test_investigate_is_non_disclosing_for_unknown_order_and_wrong_customer(
    tmp_path: Path,
) -> None:
    database = _seed_database(tmp_path, status="processing")
    repository = OrderRepository(database)
    assert repository.investigate("ORD-MISSING", "CUS-1") is None
    assert repository.investigate("ORD-1", "CUS-OTHER") is None


def test_cancel_requires_expected_order_version(tmp_path: Path) -> None:
    database = _seed_database(tmp_path, status="processing")
    repository = OrderRepository(database)
    with pytest.raises(ValueError, match="version conflict"):
        repository.cancel("ORD-1", "CUS-1", expected_version=2, review_id="review-1")


def test_cancel_is_idempotent_for_same_review(tmp_path: Path) -> None:
    database = _seed_database(tmp_path, status="processing")
    repository = OrderRepository(database)
    first = repository.cancel("ORD-1", "CUS-1", expected_version=1, review_id="review-1")
    replay = repository.cancel("ORD-1", "CUS-1", expected_version=1, review_id="review-1")
    assert first.status == "cancelled"
    assert replay.status == "cancelled"
    assert replay.replayed is True
    assert first.replayed is False


def test_cancel_sets_status_cancelled_at_and_increments_version(tmp_path: Path) -> None:
    database = _seed_database(tmp_path, status="processing")
    repository = OrderRepository(database)
    repository.cancel("ORD-1", "CUS-1", expected_version=1, review_id="review-1")
    investigation = repository.investigate("ORD-1", "CUS-1")
    assert investigation is not None
    assert investigation.status == "cancelled"
    assert investigation.version == 2
    with database.connect() as c:
        row = c.execute("SELECT cancelled_at FROM orders WHERE order_id='ORD-1'").fetchone()
        audit = c.execute(
            "SELECT COUNT(*) n FROM retail_audit_events WHERE event_type='order_cancelled'"
        ).fetchone()
    assert row["cancelled_at"] is not None
    assert audit["n"] == 1


def test_list_for_customer_returns_only_that_customers_orders_most_recent_first(
    tmp_path: Path,
) -> None:
    database = _seed_database(tmp_path, status="processing")
    with database.connect() as c:
        c.execute(
            "INSERT INTO customers (customer_id,email) VALUES (?,?)",
            ("CUS-2", "other@example.test"),
        )
        c.execute(
            "INSERT INTO orders ("
            "order_id,customer_id,status,total,currency,delivered_at,"
            "payment_status,fulfilment_status,shipped_at,cancelled_at,version"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                "ORD-2",
                "CUS-2",
                "processing",
                "50",
                "EUR",
                None,
                "unpaid",
                "processing",
                None,
                None,
                1,
            ),
        )
        c.execute(
            "INSERT INTO orders ("
            "order_id,customer_id,status,total,currency,delivered_at,"
            "payment_status,fulfilment_status,shipped_at,cancelled_at,version"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("ORD-3", "CUS-1", "delivered", "30", "EUR", None, "paid", "delivered", None, None, 1),
        )
    repository = OrderRepository(database)
    orders = repository.list_for_customer("CUS-1")
    assert [o.order_id for o in orders] == ["ORD-3", "ORD-1"]


def test_list_active_offers_joins_offers_to_products_and_computes_price_with_decimal(
    tmp_path: Path,
) -> None:
    db = _catalogue_database(tmp_path)
    repo = ProductRepository(db)
    offers = repo.list_active_offers(limit=10)
    assert {offer.product_id for offer in offers} == {"P-OFFER-1", "P-OFFER-2"}
    rain_jacket_offer = next(o for o in offers if o.product_id == "P-OFFER-1")
    assert rain_jacket_offer.list_price == Decimal("100.00")
    assert rain_jacket_offer.discount_percent == Decimal("10")
    assert rain_jacket_offer.effective_price == Decimal("90.00")
    assert rain_jacket_offer.currency == "EUR"


def test_effective_price_computes_discounted_price_and_is_none_without_an_offer(
    tmp_path: Path,
) -> None:
    db = _catalogue_database(tmp_path)
    repo = ProductRepository(db)
    offer = repo.effective_price("P-OFFER-2")
    assert offer is not None
    assert offer.effective_price == Decimal("64.00")
    assert repo.effective_price("P-PLAIN-1") is None
    assert repo.effective_price("P-DOES-NOT-EXIST") is None


def test_recommend_returns_only_database_facts(tmp_path: Path) -> None:
    db = _catalogue_database(tmp_path)
    repo = ProductRepository(db)
    recommendations = repo.recommend("jacket", category=None, max_price=None, limit=5)
    names = [r.product.name for r in recommendations]
    assert "Rain Jacket" in names
    assert "Winter Jacket" in names
    assert "Summer Jacket" not in names  # excluded: zero stock
    assert all(r.product.stock > 0 for r in recommendations)
    top = recommendations[0]
    assert top.rank == 1
    assert top.product.name == "Rain Jacket"
    # The match reason is built from stored facts, not invented ones.
    assert str(top.product.stock) in top.match_reason
    assert "10" in top.match_reason

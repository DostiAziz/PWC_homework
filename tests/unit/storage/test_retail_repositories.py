from decimal import Decimal

import pytest

from domain.models import CancellationPreview
from storage.database import Database
from storage.retail_repositories import (
    CancellationConflict,
    OrderRepository,
    ProductRepository,
)


def test_active_offers_include_database_computed_effective_price(retail_db: Database) -> None:
    offers = ProductRepository(retail_db).list_active_offers(query="jacket")

    assert len(offers) == 1
    assert offers[0].product_id == "PROD-1001"
    assert offers[0].name == "Trail Shell"
    assert offers[0].list_price == Decimal("129.99")
    assert offers[0].discount_percent == Decimal("10")
    assert offers[0].effective_price == Decimal("116.99")


def test_product_search_finds_by_name_and_category(retail_db: Database) -> None:
    repo = ProductRepository(retail_db)
    jackets = repo.search("jacket")
    assert len(jackets) >= 2
    names = [j.name for j in jackets]
    assert "Trail Shell" in names
    assert "City Parka" in names


def test_order_lookup_is_customer_scoped(retail_db: Database) -> None:
    orders = OrderRepository(retail_db)

    assert orders.lookup("ORD-2001", "CUS-1001") is not None
    assert orders.lookup("ORD-2001", "CUS-1002") is None


def test_cancel_requires_expected_order_version(retail_db: Database) -> None:
    repository = OrderRepository(retail_db)
    preview = CancellationPreview(
        confirmation_token="token-123",
        order_id="ORD-2001",
        customer_id="CUS-1001",
        expected_version=2,
        summary="Cancel ORD-2001",
    )
    with pytest.raises(CancellationConflict, match="order changed before cancellation"):
        repository.cancel(preview)


def test_cancel_is_idempotent_for_same_token(retail_db: Database) -> None:
    repository = OrderRepository(retail_db)
    preview = CancellationPreview(
        confirmation_token="token-123",
        order_id="ORD-2001",
        customer_id="CUS-1001",
        expected_version=1,
        summary="Cancel ORD-2001",
    )
    first = repository.cancel(preview)
    replay = repository.cancel(preview)
    assert first.status == "cancelled"
    assert replay.status == "cancelled"
    assert replay.replayed is True
    assert first.replayed is False


def test_cancel_sets_status_cancelled_at_and_increments_version(retail_db: Database) -> None:
    repository = OrderRepository(retail_db)
    preview = CancellationPreview(
        confirmation_token="token-123",
        order_id="ORD-4001",
        customer_id="CUS-1001",
        expected_version=1,
        summary="Cancel ORD-4001",
    )
    repository.cancel(preview)
    order = repository.lookup("ORD-4001", "CUS-1001")
    assert order is not None
    assert order.status == "cancelled"
    assert order.version == 2
    with retail_db.connect() as c:
        row = c.execute("SELECT cancelled_at FROM orders WHERE order_id='ORD-4001'").fetchone()
        action = c.execute(
            "SELECT COUNT(*) n FROM cancellation_actions WHERE confirmation_token='token-123'"
        ).fetchone()

    assert row["cancelled_at"] is not None
    assert action["n"] == 1

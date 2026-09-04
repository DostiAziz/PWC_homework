from datetime import UTC, datetime, timedelta
from pathlib import Path

from pwc_support.storage.database import Database
from pwc_support.storage.retail_repositories import (
    OrderRepository,
    ProductRepository,
    ReturnRepository,
)
from pwc_support.workflow.graph import build_graph
from pwc_support.workflow.retail_tools import build_retail_tools


def _db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "retail.sqlite3")
    db.initialize()
    delivered = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    with db.connect() as c:
        c.execute(
            "INSERT INTO products (product_id,name,category,price,currency,attributes_json,active) "
            "VALUES (?,?,?,?,?,?,?)",
            ("P-1", "Trail Jacket", "outerwear", "120", "EUR", "{}", 1),
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
            (
                "ORD-1",
                "C-1",
                "delivered",
                "120",
                "EUR",
                delivered,
                "paid",
                "delivered",
                delivered,
                None,
                1,
            ),
        )
        c.execute(
            "INSERT INTO order_items VALUES (?,?,?,?,?)", ("ITEM-1", "ORD-1", "P-1", 1, "120")
        )
    return db


def test_product_order_and_exception_paths_are_distinct(tmp_path: Path) -> None:
    database = _db(tmp_path)
    tools = build_retail_tools(
        ProductRepository(database), OrderRepository(database), "C-1", ReturnRepository(database)
    )
    graph = build_graph(retail_tools=tools)
    product = graph.invoke({"message": {"body": "Show me a jacket", "sender_id": "C-1"}})
    order = graph.invoke(
        {"message": {"body": "What is the status of order ORD-1?", "sender_id": "C-1"}}
    )
    exception = graph.invoke(
        {
            "message": {
                "body": "I want a refund for ORD-1 ITEM-1 because it is damaged",
                "sender_id": "C-1",
            }
        }
    )
    assert product["outcome"]["status"] == "answered" and product["outcome"]["case_id"] is None
    assert order["outcome"]["status"] == "answered" and "delivered" in order["delivery"]["message"]
    assert (
        exception["outcome"]["status"] == "pending_review"
        and exception["review_request"]["proposed_actions"]
    )

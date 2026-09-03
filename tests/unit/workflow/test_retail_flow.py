from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from pwc_support.storage.database import Database
from pwc_support.storage.retail_repositories import OrderRepository, ProductRepository, ReturnRepository
from pwc_support.workflow.retail_actions import request_return
from pwc_support.workflow.retail_tools import build_retail_tools
from pwc_support.workflow.graph import build_graph


def _db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "retail.sqlite3")
    db.initialize()
    with db.connect() as c:
        c.execute("INSERT INTO products VALUES (?,?,?,?,?,?,?)", ("P-1", "Trail Jacket", "outerwear", "129.00", "EUR", '{}', 1))
        c.execute("INSERT INTO inventory VALUES (?,?,?)", ("P-1", "WH-1", 4))
        c.execute("INSERT INTO customers VALUES (?,?)", ("CUS-1", "a@example.test"))
        delivered = (datetime.now(UTC) - timedelta(days=2)).isoformat()
        c.execute("INSERT INTO orders VALUES (?,?,?,?,?,?)", ("ORD-1", "CUS-1", "delivered", "129.00", "EUR", delivered))
        c.execute("INSERT INTO order_items VALUES (?,?,?,?,?)", ("ITEM-1", "ORD-1", "P-1", 1, "129.00"))
    return db


def test_product_tool_returns_database_facts(tmp_path: Path) -> None:
    db = _db(tmp_path)
    tools = build_retail_tools(ProductRepository(db), OrderRepository(db), "CUS-1")
    result = tools[0].invoke({"query": "jacket"})
    assert result["products"][0]["name"] == "Trail Jacket"
    assert result["products"][0]["stock"] == 4


def test_eligible_return_is_idempotent_and_creates_pending_review(tmp_path: Path) -> None:
    db = _db(tmp_path)
    repo = ReturnRepository(db)
    first, eligibility = request_return(repo, order_id="ORD-1", item_id="ITEM-1", reason="wrong size", idempotency_key="idem-1")
    second, _ = request_return(repo, order_id="ORD-1", item_id="ITEM-1", reason="wrong size", idempotency_key="idem-1")
    assert eligibility.eligible is True
    assert first is not None and first.status == "pending_review"
    assert second is not None and second.return_id == first.return_id


def test_graph_answers_product_question_without_case(tmp_path: Path) -> None:
    db = _db(tmp_path)
    result = build_graph(retail_tools=build_retail_tools(ProductRepository(db), OrderRepository(db), "CUS-1", ReturnRepository(db))).invoke({"message": {"body": "What jacket do you have?"}})
    assert result["outcome"]["status"] == "answered"
    assert "Trail Jacket" in result["delivery"]["message"]
    assert result["outcome"].get("case_id") is None

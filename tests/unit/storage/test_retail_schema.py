from pathlib import Path

from retail_support.storage.database import Database


def test_schema_contains_only_current_business_tables(tmp_path: Path) -> None:
    database = Database(tmp_path / "schema.sqlite3")
    database.initialize()
    with database.connect() as connection:
        tables = {
            row["name"]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }

    assert {
        "products",
        "inventory",
        "offers",
        "customers",
        "orders",
        "cancellation_actions",
    } <= tables
    assert {
        "cases",
        "review_requests",
        "outbox_messages",
        "return_requests",
        "refund_requests",
        "order_cancellation_actions",
    }.isdisjoint(tables)

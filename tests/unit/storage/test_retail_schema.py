from pathlib import Path

from pwc_support.storage.database import Database


def test_retail_schema_contains_all_operational_tables(tmp_path: Path) -> None:
    database = Database(tmp_path / "schema.sqlite3")
    database.initialize()
    with database.connect() as connection:
        tables = {row["name"] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"products", "inventory", "offers", "customers", "orders", "order_items", "return_requests", "refund_requests", "retail_audit_events"} <= tables


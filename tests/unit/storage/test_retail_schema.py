import sqlite3
from pathlib import Path

from pwc_support.storage.database import Database


def test_retail_schema_contains_all_operational_tables(tmp_path: Path) -> None:
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
        "order_items",
        "return_requests",
        "refund_requests",
        "retail_audit_events",
        "order_cancellation_actions",
    } <= tables


def test_initialize_migrates_a_pre_task_2_orders_table_additively(tmp_path: Path) -> None:
    """A database written before Task 2 has the old, narrower `orders` schema and no
    `order_cancellation_actions` table. `Database.initialize()` must backfill the new
    columns and create the new table without disturbing the pre-existing row."""
    path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE customers (customer_id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE);
            CREATE TABLE orders (
                order_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL REFERENCES customers(customer_id),
                status TEXT NOT NULL,
                total NUMERIC NOT NULL CHECK(total >= 0),
                currency TEXT NOT NULL,
                delivered_at TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO customers (customer_id, email) VALUES (?, ?)",
            ("CUS-OLD", "old@example.test"),
        )
        connection.execute(
            "INSERT INTO orders (order_id, customer_id, status, total, currency, delivered_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("ORD-OLD", "CUS-OLD", "delivered", "10.00", "EUR", None),
        )
        connection.commit()
    finally:
        connection.close()

    database = Database(path)
    database.initialize()

    with database.connect() as c:
        columns = {row["name"] for row in c.execute("PRAGMA table_info(orders)").fetchall()}
        row = c.execute("SELECT * FROM orders WHERE order_id='ORD-OLD'").fetchone()
        tables = {
            table_row["name"]
            for table_row in c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }

    assert {
        "payment_status",
        "fulfilment_status",
        "shipped_at",
        "cancelled_at",
        "version",
    } <= columns
    assert row["order_id"] == "ORD-OLD"
    assert row["status"] == "delivered"
    assert row["payment_status"] == "unpaid"
    assert row["fulfilment_status"] == "processing"
    assert row["shipped_at"] is None
    assert row["cancelled_at"] is None
    assert row["version"] == 1
    assert "order_cancellation_actions" in tables

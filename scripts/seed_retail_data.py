from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pwc_support.config import Settings
from pwc_support.storage.database import Database


def seed(database: Database) -> None:
    now = datetime.now(UTC)
    with database.connect() as c:
        c.executemany(
            "INSERT OR IGNORE INTO products "
            "(product_id,name,category,price,currency,attributes_json,active) "
            "VALUES (?,?,?,?,?,?,?)",
            [
                (
                    "PROD-1001",
                    "Trail Shell",
                    "jackets",
                    "129.99",
                    "EUR",
                    json.dumps({"waterproof": True, "colour": "navy"}),
                    1,
                ),
                (
                    "PROD-1002",
                    "City Parka",
                    "jackets",
                    "99.99",
                    "EUR",
                    json.dumps({"waterproof": True, "colour": "black"}),
                    1,
                ),
                (
                    "PROD-2001",
                    "Travel Backpack",
                    "bags",
                    "79.99",
                    "EUR",
                    json.dumps({"capacity_l": 28}),
                    1,
                ),
                (
                    "PROD-3001",
                    "Noise Cancel Headphones",
                    "electronics",
                    "149.99",
                    "EUR",
                    json.dumps({"wireless": True}),
                    1,
                ),
                (
                    "PROD-3002",
                    "Studio Earbuds",
                    "electronics",
                    "59.99",
                    "EUR",
                    json.dumps({"wireless": True}),
                    1,
                ),
                (
                    "PROD-4001",
                    "Merino Base Layer",
                    "clothing",
                    "49.99",
                    "EUR",
                    json.dumps({"warmth": "medium"}),
                    1,
                ),
                (
                    "PROD-4002",
                    "Everyday T-Shirt",
                    "clothing",
                    "19.99",
                    "EUR",
                    json.dumps({"colour": "white"}),
                    1,
                ),
                (
                    "PROD-5001",
                    "Commuter Shoes",
                    "footwear",
                    "89.99",
                    "EUR",
                    json.dumps({"waterproof": True}),
                    1,
                ),
                (
                    "PROD-5002",
                    "Trail Socks",
                    "footwear",
                    "14.99",
                    "EUR",
                    json.dumps({"material": "wool"}),
                    1,
                ),
                (
                    "PROD-6001",
                    "Travel Mug",
                    "accessories",
                    "24.99",
                    "EUR",
                    json.dumps({"capacity_ml": 350}),
                    1,
                ),
                (
                    "PROD-7001",
                    "Expedition Kit",
                    "equipment",
                    "999.99",
                    "EUR",
                    json.dumps({"waterproof": True, "professional": True}),
                    1,
                ),
            ],
        )
        c.executemany(
            "INSERT OR IGNORE INTO inventory (product_id,location,quantity) VALUES (?,?,?)",
            [
                ("PROD-1001", "budapest", 7),
                ("PROD-1001", "vienna", 4),
                ("PROD-1002", "budapest", 3),
                ("PROD-2001", "budapest", 12),
                ("PROD-3001", "budapest", 0),
                ("PROD-3002", "budapest", 8),
                ("PROD-4001", "budapest", 20),
                ("PROD-4002", "budapest", 30),
                ("PROD-5001", "budapest", 5),
                ("PROD-5002", "budapest", 40),
                ("PROD-6001", "budapest", 10),
                ("PROD-7001", "budapest", 1),
            ],
        )
        c.executemany(
            "INSERT OR IGNORE INTO offers "
            "(offer_id,product_id,description,discount_percent,active) VALUES (?,?,?,?,?)",
            [
                ("OFFER-10", "PROD-1001", "10% off", 10, 1),
                ("OFFER-OLD", "PROD-1002", "5% off expired", 5, 0),
                ("OFFER-EXP", "PROD-7001", "15% off", 15, 1),
            ],
        )
        c.execute(
            "INSERT OR IGNORE INTO customers (customer_id,email) VALUES (?,?)",
            ("CUS-1001", "client@example.test"),
        )
        c.execute(
            "INSERT OR IGNORE INTO customers (customer_id,email) VALUES (?,?)",
            ("CUS-1002", "late@example.test"),
        )
        order_columns = (
            "order_id,customer_id,status,total,currency,delivered_at,"
            "payment_status,fulfilment_status,shipped_at,cancelled_at,version"
        )
        # ORD-1001 is the standard client fixture: a fully delivered order.
        c.execute(
            f"INSERT OR IGNORE INTO orders ({order_columns}) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                "ORD-1001",
                "CUS-1001",
                "delivered",
                "129.99",
                "EUR",
                (now - timedelta(days=5)).isoformat(),
                "paid",
                "delivered",
                (now - timedelta(days=8)).isoformat(),
                None,
                1,
            ),
        )
        c.execute(
            "INSERT OR IGNORE INTO order_items (item_id,order_id,product_id,quantity,unit_price) "
            "VALUES (?,?,?,?,?)",
            ("ITEM-1001", "ORD-1001", "PROD-1001", 1, "129.99"),
        )
        # ORD-2001: processing and unpaid.
        c.execute(
            f"INSERT OR IGNORE INTO orders ({order_columns}) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                "ORD-2001",
                "CUS-1001",
                "processing",
                "79.99",
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
            "INSERT OR IGNORE INTO order_items (item_id,order_id,product_id,quantity,unit_price) "
            "VALUES (?,?,?,?,?)",
            ("ITEM-2001", "ORD-2001", "PROD-2001", 1, "79.99"),
        )
        # ORD-3001: delivered, but well outside the return window.
        c.execute(
            f"INSERT OR IGNORE INTO orders ({order_columns}) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                "ORD-3001",
                "CUS-1002",
                "delivered",
                "999.99",
                "EUR",
                (now - timedelta(days=45)).isoformat(),
                "paid",
                "delivered",
                (now - timedelta(days=48)).isoformat(),
                None,
                1,
            ),
        )
        c.execute(
            "INSERT OR IGNORE INTO order_items (item_id,order_id,product_id,quantity,unit_price) "
            "VALUES (?,?,?,?,?)",
            ("ITEM-3001", "ORD-3001", "PROD-7001", 1, "999.99"),
        )
        # ORD-4001: paid, but not yet shipped.
        c.execute(
            f"INSERT OR IGNORE INTO orders ({order_columns}) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                "ORD-4001",
                "CUS-1001",
                "paid",
                "59.99",
                "EUR",
                None,
                "paid",
                "processing",
                None,
                None,
                1,
            ),
        )
        c.execute(
            "INSERT OR IGNORE INTO order_items (item_id,order_id,product_id,quantity,unit_price) "
            "VALUES (?,?,?,?,?)",
            ("ITEM-4001", "ORD-4001", "PROD-3002", 1, "59.99"),
        )
        # ORD-5001: paid and shipped, but not yet delivered.
        c.execute(
            f"INSERT OR IGNORE INTO orders ({order_columns}) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                "ORD-5001",
                "CUS-1001",
                "shipped",
                "49.99",
                "EUR",
                None,
                "paid",
                "shipped",
                (now - timedelta(days=1)).isoformat(),
                None,
                1,
            ),
        )
        c.execute(
            "INSERT OR IGNORE INTO order_items (item_id,order_id,product_id,quantity,unit_price) "
            "VALUES (?,?,?,?,?)",
            ("ITEM-5001", "ORD-5001", "PROD-4001", 1, "49.99"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Settings.from_env().retail_db)
    args = parser.parse_args()
    database = Database(args.db)
    database.initialize()
    seed(database)

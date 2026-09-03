from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from pwc_support.storage.database import Database


def seed(database: Database) -> None:
    now = datetime.now(UTC)
    with database.connect() as c:
        c.executemany("INSERT OR IGNORE INTO products VALUES (?,?,?,?,?,?,?)", [
            ("PROD-1001", "Trail Shell", "jackets", "129.99", "EUR", json.dumps({"waterproof": True, "colour": "navy"}), 1),
            ("PROD-1002", "City Parka", "jackets", "99.99", "EUR", json.dumps({"waterproof": True, "colour": "black"}), 1),
            ("PROD-2001", "Travel Backpack", "bags", "79.99", "EUR", json.dumps({"capacity_l": 28}), 1),
            ("PROD-3001", "Noise Cancel Headphones", "electronics", "149.99", "EUR", json.dumps({"wireless": True}), 1),
        ])
        c.executemany("INSERT OR IGNORE INTO inventory VALUES (?,?,?)", [("PROD-1001", "budapest", 7), ("PROD-1002", "budapest", 3), ("PROD-2001", "budapest", 12), ("PROD-3001", "budapest", 0)])
        c.execute("INSERT OR IGNORE INTO offers VALUES (?,?,?,?,?)", ("OFFER-10", "PROD-1001", "10% off", 10, 1))
        c.execute("INSERT OR IGNORE INTO customers VALUES (?,?)", ("CUS-1001", "client@example.test"))
        c.execute("INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?)", ("ORD-1001", "CUS-1001", "delivered", "129.99", "EUR", (now - timedelta(days=5)).isoformat()))
        c.execute("INSERT OR IGNORE INTO order_items VALUES (?,?,?,?,?)", ("ITEM-1001", "ORD-1001", "PROD-1001", 1, "129.99"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("data/state/retail.sqlite3"))
    args = parser.parse_args()
    database = Database(args.db)
    database.initialize()
    seed(database)

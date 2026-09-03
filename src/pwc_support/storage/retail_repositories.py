from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from pwc_support.domain.errors import ErrorCode, SupportError
from pwc_support.domain.models import OrderSummary, ProductSummary, ReturnEligibility, ReturnRequest, RetailActionRisk
from pwc_support.storage.database import Database


class ProductRepository:
    def __init__(self, database: Database) -> None: self.database = database
    def search(self, query: str, category: str | None = None, max_price: Decimal | None = None, attributes: tuple[str, ...] = ()) -> tuple[ProductSummary, ...]:
        clauses = ["active=1", "(lower(name) LIKE ? OR lower(category) LIKE ? OR lower(attributes_json) LIKE ?)"]
        params: list[object] = [f"%{query.lower()}%", f"%{query.lower()}%", f"%{query.lower()}%"]
        if category: clauses.append("category=?"); params.append(category)
        if max_price is not None: clauses.append("price<=?"); params.append(str(max_price))
        with self.database.connect() as c:
            rows = c.execute(f"SELECT * FROM products WHERE {' AND '.join(clauses)} ORDER BY price LIMIT 20", params).fetchall()
            return tuple(self._product(c, row) for row in rows if all(a.lower() in row["attributes_json"].lower() for a in attributes))
    def get(self, product_id: str) -> ProductSummary | None:
        with self.database.connect() as c:
            row = c.execute("SELECT * FROM products WHERE product_id=? AND active=1", (product_id,)).fetchone()
            return self._product(c, row) if row else None
    def _product(self, c: object, row: sqlite3.Row) -> ProductSummary:
        quantity = 0
        with self.database.connect() as cx:
            inv = cx.execute("SELECT COALESCE(SUM(quantity),0) q FROM inventory WHERE product_id=?", (row["product_id"],)).fetchone()
            quantity = int(inv["q"])
            offer = cx.execute("SELECT offer_id,description,discount_percent FROM offers WHERE product_id=? AND active=1 LIMIT 1", (row["product_id"],)).fetchone()
        return ProductSummary(product_id=row["product_id"], name=row["name"], category=row["category"], price=Decimal(str(row["price"])), currency=row["currency"], stock=quantity, attributes=json.loads(row["attributes_json"]), active_offer=dict(offer) if offer else None)


class OrderRepository:
    def __init__(self, database: Database) -> None: self.database = database
    def lookup(self, order_id: str, customer_id: str) -> OrderSummary | None:
        with self.database.connect() as c:
            row = c.execute("SELECT * FROM orders WHERE order_id=? AND customer_id=?", (order_id, customer_id)).fetchone()
            if not row: return None
            items = c.execute("SELECT item_id,product_id,quantity,unit_price FROM order_items WHERE order_id=?", (order_id,)).fetchall()
        return OrderSummary(order_id=row["order_id"], customer_id=row["customer_id"], status=row["status"], total=Decimal(str(row["total"])), currency=row["currency"], items=tuple(dict(item) for item in items), delivered_at=datetime.fromisoformat(row["delivered_at"]) if row["delivered_at"] else None)


class ReturnRepository:
    def __init__(self, database: Database, window_days: int = 30) -> None: self.database, self.window_days = database, window_days
    def evaluate(self, order_id: str, item_id: str, reason: str, now: datetime) -> ReturnEligibility:
        with self.database.connect() as c:
            row = c.execute("SELECT o.delivered_at,oi.unit_price FROM orders o JOIN order_items oi ON oi.order_id=o.order_id WHERE o.order_id=? AND oi.item_id=?", (order_id,item_id)).fetchone()
        if not row: return ReturnEligibility(eligible=False, reason="Order item was not found", refund_amount=Decimal("0"), risk_flags=(RetailActionRisk.POLICY_EXCEPTION,))
        delivered = datetime.fromisoformat(row["delivered_at"]) if row["delivered_at"] else None
        amount = Decimal(str(row["unit_price"]))
        if delivered is None or now - delivered > timedelta(days=self.window_days):
            return ReturnEligibility(eligible=False, reason="Outside return window", deadline=delivered + timedelta(days=self.window_days) if delivered else None, refund_amount=amount, risk_flags=(RetailActionRisk.POLICY_EXCEPTION,))
        return ReturnEligibility(eligible=True, reason="Within return window", deadline=delivered + timedelta(days=self.window_days), refund_amount=amount, risk_flags=())
    def create(self, request: ReturnRequest) -> ReturnRequest:
        with self.database.connect() as c:
            existing = c.execute("SELECT * FROM return_requests WHERE idempotency_key=?", (request.idempotency_key,)).fetchone()
            if existing: return ReturnRequest(return_id=existing["return_id"], order_id=existing["order_id"], item_id=existing["item_id"], reason=existing["reason"], status=existing["status"], idempotency_key=existing["idempotency_key"])
            c.execute("INSERT INTO return_requests VALUES (?,?,?,?,?,?,?)", (request.return_id,request.order_id,request.item_id,request.reason,request.status,request.idempotency_key,datetime.now(UTC).isoformat()))
        return request

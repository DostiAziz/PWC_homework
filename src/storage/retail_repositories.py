from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from decimal import Decimal

from domain.models import (
    CancellationPreview,
    CancellationResult,
    OfferSummary,
    OrderSummary,
    ProductSummary,
)
from storage.database import Database


class CancellationConflict(Exception):
    pass


class ProductRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def search(self, query: str, limit: int = 20, **_: object) -> tuple[ProductSummary, ...]:
        stripped = query.strip()
        with self.database.connect() as connection:
            if not stripped:
                # Empty query: return all active products
                rows = connection.execute(
                    "SELECT p.product_id, p.name, p.category, p.price, p.currency, "
                    "COALESCE(SUM(i.quantity), 0) AS stock "
                    "FROM products p LEFT JOIN inventory i ON i.product_id = p.product_id "
                    "WHERE p.active = 1 "
                    "GROUP BY p.product_id, p.name, p.category, p.price, p.currency "
                    "ORDER BY p.price LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                needle = f"%{stripped.casefold()}%"
                rows = connection.execute(
                    "SELECT p.product_id, p.name, p.category, p.price, p.currency, "
                    "COALESCE(SUM(i.quantity), 0) AS stock "
                    "FROM products p LEFT JOIN inventory i ON i.product_id = p.product_id "
                    "WHERE p.active = 1 AND "
                    "(lower(p.name) LIKE ? OR lower(p.category) LIKE ?) "
                    "GROUP BY p.product_id, p.name, p.category, p.price, p.currency "
                    "ORDER BY p.price LIMIT ?",
                    (needle, needle, limit),
                ).fetchall()
        return tuple(ProductSummary.model_validate(dict(row)) for row in rows)

    def list_active_offers(
        self, query: str | None = None, limit: int = 20
    ) -> tuple[OfferSummary, ...]:
        clauses = ["o.active = 1", "p.active = 1"]
        params: list[object] = []
        if query:
            clauses.append("(lower(p.name) LIKE ? OR lower(p.category) LIKE ?)")
            needle = f"%{query.casefold()}%"
            params.extend((needle, needle))
        params.append(limit)
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT o.offer_id, o.product_id, o.description, o.discount_percent, "
                "p.name, p.price, p.currency FROM offers o "
                "JOIN products p ON p.product_id = o.product_id "
                f"WHERE {' AND '.join(clauses)} ORDER BY o.offer_id LIMIT ?",
                params,
            ).fetchall()
        return tuple(self._offer_summary(row) for row in rows)

    @staticmethod
    def _offer_summary(row: sqlite3.Row) -> OfferSummary:
        list_price = Decimal(str(row["price"]))
        discount = Decimal(str(row["discount_percent"]))
        effective = (list_price * (Decimal("1") - discount / Decimal("100"))).quantize(
            Decimal("0.01")
        )
        return OfferSummary(
            offer_id=str(row["offer_id"]),
            product_id=str(row["product_id"]),
            name=str(row["name"]),
            description=str(row["description"]),
            list_price=list_price,
            discount_percent=discount,
            effective_price=effective,
            currency=str(row["currency"]),
        )


class OrderRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def list_orders(self, customer_id: str, status: str | None = None) -> tuple[OrderSummary, ...]:
        with self.database.connect() as connection:
            if status:
                rows = connection.execute(
                    "SELECT order_id, customer_id, status, total, currency, "
                    "fulfilment_status, version FROM orders "
                    "WHERE customer_id = ? AND status = ? ORDER BY order_id",
                    (customer_id, status),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT order_id, customer_id, status, total, currency, "
                    "fulfilment_status, version FROM orders "
                    "WHERE customer_id = ? ORDER BY order_id",
                    (customer_id,),
                ).fetchall()

        return tuple(
            OrderSummary(
                order_id=row["order_id"],
                customer_id=row["customer_id"],
                status=row["status"],
                total=Decimal(str(row["total"])),
                currency=row["currency"],
                fulfilment_status=row["fulfilment_status"],
                version=row["version"],
            )
            for row in rows
        )

    def lookup(self, order_id: str, customer_id: str) -> OrderSummary | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT order_id, customer_id, status, total, currency, "
                "fulfilment_status, version FROM orders "
                "WHERE order_id = ? AND customer_id = ?",
                (order_id, customer_id),
            ).fetchone()
        if row is None:
            return None
        return OrderSummary.model_validate(dict(row))

    def cancel(self, preview: CancellationPreview) -> CancellationResult:
        now = datetime.now(UTC).isoformat()
        connection = self.database.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            action = connection.execute(
                "SELECT order_id, customer_id, status FROM cancellation_actions "
                "WHERE confirmation_token = ?",
                (preview.confirmation_token,),
            ).fetchone()
            if action:
                if (
                    action["order_id"] != preview.order_id
                    or action["customer_id"] != preview.customer_id
                ):
                    raise CancellationConflict("token mismatch for order or customer")
                return CancellationResult(
                    order_id=preview.order_id,
                    status="cancelled",
                    replayed=True,
                )
            cursor = connection.execute(
                "UPDATE orders SET status = 'cancelled', cancelled_at = ?, version = version + 1 "
                "WHERE order_id = ? AND customer_id = ? AND version = ? "
                "  AND total = ? AND currency = ? "
                "  AND status IN ('processing', 'paid') AND fulfilment_status = 'processing'",
                (
                    now,
                    preview.order_id,
                    preview.customer_id,
                    preview.expected_version,
                    float(preview.expected_total),
                    preview.expected_currency,
                ),
            )
            if cursor.rowcount == 0:
                raise CancellationConflict("order changed before cancellation")
            connection.execute(
                "INSERT INTO cancellation_actions "
                "(confirmation_token, order_id, customer_id, status, created_at, completed_at) "
                "VALUES (?, ?, ?, 'cancelled', ?, ?)",
                (preview.confirmation_token, preview.order_id, preview.customer_id, now, now),
            )
            items = connection.execute(
                "SELECT product_id, quantity FROM order_items WHERE order_id = ?",
                (preview.order_id,)
            ).fetchall()
            for item in items:
                connection.execute(
                    "UPDATE inventory SET quantity = quantity + ? "
                    "WHERE product_id = ? AND location = 'budapest'",
                    (item["quantity"], item["product_id"])
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

        return CancellationResult(
            order_id=preview.order_id,
            status="cancelled",
            replayed=False,
        )

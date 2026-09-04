from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from pwc_support.domain.models import (
    CancellationExecutionResult,
    OfferSummary,
    OrderInvestigation,
    OrderSummary,
    ProductRecommendation,
    ProductSummary,
    RetailActionRisk,
    ReturnEligibility,
    ReturnRequest,
)
from pwc_support.storage.database import Database


class ProductRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def search(self, query: str, limit: int = 20, **_: object) -> tuple[ProductSummary, ...]:
        needle = f"%{query.casefold()}%"
        with self.database.connect() as connection:
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

    def get(self, product_id: str) -> ProductSummary | None:
        with self.database.connect() as c:
            row = c.execute(
                "SELECT * FROM products WHERE product_id=? AND active=1", (product_id,)
            ).fetchone()
            return self._product(c, row) if row else None

    def inventory(self, product_id: str, location: str | None = None) -> int:
        query = "SELECT COALESCE(SUM(quantity),0) q FROM inventory WHERE product_id=?"
        params: list[object] = [product_id]
        if location:
            query += " AND location=?"
            params.append(location)
        with self.database.connect() as c:
            row = c.execute(query, params).fetchone()
        return int(row["q"] if row else 0)

    def offer(self, product_id: str) -> dict[str, object] | None:
        with self.database.connect() as c:
            row = c.execute(
                "SELECT offer_id,description,discount_percent FROM offers "
                "WHERE product_id=? AND active=1 ORDER BY offer_id LIMIT 1",
                (product_id,),
            ).fetchone()
        return dict(row) if row else None

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

    def effective_price(self, product_id: str) -> OfferSummary | None:
        """The database-computed discounted price for a product's active offer, if any."""
        with self.database.connect() as c:
            row = c.execute(
                "SELECT o.offer_id, o.product_id, o.description, "
                "o.discount_percent, p.name, p.price, p.currency "
                "FROM offers o JOIN products p ON p.product_id = o.product_id "
                "WHERE o.product_id = ? AND o.active = 1 AND p.active = 1 "
                "ORDER BY o.offer_id LIMIT 1",
                (product_id,),
            ).fetchone()
        return self._offer_summary(row) if row else None

    @staticmethod
    def _offer_summary(row: sqlite3.Row) -> OfferSummary:
        list_price = Decimal(str(row["price"]))
        discount_percent = Decimal(str(row["discount_percent"]))
        effective_price = (
            list_price * (Decimal(1) - discount_percent / Decimal(100))
        ).quantize(Decimal("0.01"))
        return OfferSummary(
            offer_id=row["offer_id"],
            product_id=row["product_id"],
            name=row["name"],
            description=row["description"],
            list_price=list_price,
            discount_percent=discount_percent,
            effective_price=effective_price,
            currency=row["currency"],
        )

    def recommend(
        self,
        query: str,
        *,
        category: str | None = None,
        max_price: Decimal | None = None,
        limit: int = 5,
    ) -> tuple[ProductRecommendation, ...]:
        """In-stock catalogue matches, ranked deterministically, reasoned from stored facts."""
        candidates = [
            product
            for product in self.search(query, limit=limit)
            if product.stock > 0
        ]

        needle = query.strip().lower()

        def relevance(product: ProductSummary) -> int:
            if needle and needle in product.name.lower():
                return 0
            if needle and needle in product.category.lower():
                return 1
            return 2

        ranked = sorted(candidates, key=lambda product: (relevance(product), product.price))
        recommendations = []
        for rank, product in enumerate(ranked[:limit], start=1):
            recommendations.append(
                ProductRecommendation(
                    product=product,
                    match_reason=self._match_reason(product, needle),
                    rank=rank,
                )
            )
        return tuple(recommendations)

    def _match_reason(self, product: ProductSummary, needle: str) -> str:
        reasons: list[str] = []
        if needle and needle in product.name.lower():
            reasons.append(f"'{product.name}' matches '{needle}'")
        else:
            reasons.append(f"'{product.name}' is in category '{product.category}'")
        offer = self.offer(product.product_id)
        if offer and offer.get("discount_percent") is not None:
            reasons.append(f"{offer['discount_percent']}% active offer")
        reasons.append(f"{product.stock} in stock")

        return "; ".join(reasons)


    def _product(self, c: object, row: sqlite3.Row) -> ProductSummary:
        quantity = 0
        with self.database.connect() as cx:
            inv = cx.execute(
                "SELECT COALESCE(SUM(quantity),0) q FROM inventory WHERE product_id=?",
                (row["product_id"],),
            ).fetchone()
            quantity = int(inv["q"])
            offer = cx.execute(
                "SELECT offer_id,description,discount_percent FROM offers "
                "WHERE product_id=? AND active=1 LIMIT 1",
                (row["product_id"],),
            ).fetchone()
        return ProductSummary(
            product_id=row["product_id"],
            name=row["name"],
            category=row["category"],
            price=Decimal(str(row["price"])),
            currency=row["currency"],
            stock=quantity,
            attributes=json.loads(row["attributes_json"]),
            active_offer=dict(offer) if offer else None,
        )


class OrderRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def lookup(self, order_id: str, customer_id: str) -> OrderSummary | None:
        with self.database.connect() as c:
            row = c.execute(
                "SELECT * FROM orders WHERE order_id=? AND customer_id=?", (order_id, customer_id)
            ).fetchone()
            if not row:
                return None
            items = c.execute(
                "SELECT item_id,product_id,quantity,unit_price FROM order_items WHERE order_id=?",
                (order_id,),
            ).fetchall()
        return OrderSummary(
            order_id=row["order_id"],
            customer_id=row["customer_id"],
            status=row["status"],
            total=Decimal(str(row["total"])),
            currency=row["currency"],
            items=tuple(dict(item) for item in items),
            delivered_at=datetime.fromisoformat(row["delivered_at"])
            if row["delivered_at"]
            else None,
        )

    def investigate(self, order_id: str, customer_id: str) -> OrderInvestigation | None:
        """Verified order facts scoped to the caller's own customer id.

        Returns None both for an unknown order and for an order owned by another
        customer, so a caller cannot distinguish "wrong customer" from "no such
        order" (non-disclosure).
        """
        with self.database.connect() as c:
            row = c.execute(
                "SELECT * FROM orders WHERE order_id=? AND customer_id=?", (order_id, customer_id)
            ).fetchone()
            if not row:
                return None
            items = c.execute(
                "SELECT item_id,product_id,quantity,unit_price FROM order_items WHERE order_id=?",
                (order_id,),
            ).fetchall()
        return OrderInvestigation(
            order_id=row["order_id"],
            customer_id=row["customer_id"],
            status=row["status"],
            total=Decimal(str(row["total"])),
            currency=row["currency"],
            payment_status=row["payment_status"],
            fulfilment_status=row["fulfilment_status"],
            shipped_at=datetime.fromisoformat(row["shipped_at"]) if row["shipped_at"] else None,
            version=row["version"],
            items=tuple(dict(item) for item in items),
        )

    def list_for_customer(self, customer_id: str, limit: int = 10) -> tuple[OrderSummary, ...]:
        """That customer's own orders, most-recent-first and bounded by ``limit``."""
        with self.database.connect() as c:
            rows = c.execute(
                "SELECT * FROM orders WHERE customer_id=? ORDER BY rowid DESC LIMIT ?",
                (customer_id, limit),
            ).fetchall()
            summaries = []
            for row in rows:
                items = c.execute(
                    "SELECT item_id,product_id,quantity,unit_price FROM order_items "
                    "WHERE order_id=?",
                    (row["order_id"],),
                ).fetchall()
                summaries.append(
                    OrderSummary(
                        order_id=row["order_id"],
                        customer_id=row["customer_id"],
                        status=row["status"],
                        total=Decimal(str(row["total"])),
                        currency=row["currency"],
                        items=tuple(dict(item) for item in items),
                        delivered_at=datetime.fromisoformat(row["delivered_at"])
                        if row["delivered_at"]
                        else None,
                    )
                )
        return tuple(summaries)

    def cancel(
        self, order_id: str, customer_id: str, *, expected_version: int, review_id: str
    ) -> CancellationExecutionResult:
        """Optimistically cancel an order, idempotent on ``review_id``.

        A replayed call with the same ``review_id`` reuses the recorded action
        instead of re-checking the version, so approving the same reviewed
        cancellation twice is safe even after the order's version has moved on.
        """
        with self.database.connect() as c:
            existing = c.execute(
                "SELECT * FROM order_cancellation_actions WHERE review_id=?", (review_id,)
            ).fetchone()
            if existing:
                return CancellationExecutionResult(
                    order_id=existing["order_id"],
                    review_id=review_id,
                    status=existing["status"],
                    replayed=True,
                )
            now = datetime.now(UTC).isoformat()
            updated = c.execute(
                "UPDATE orders SET status='cancelled', cancelled_at=?, version=version+1 "
                "WHERE order_id=? AND customer_id=? AND version=?",
                (now, order_id, customer_id, expected_version),
            ).rowcount
            if not updated:
                raise ValueError("order not found or version conflict")
            c.execute(
                "INSERT INTO order_cancellation_actions ("
                "review_id,idempotency_key,order_id,customer_id,status,created_at,completed_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (review_id, review_id, order_id, customer_id, "cancelled", now, now),
            )
            c.execute(
                "INSERT INTO retail_audit_events "
                "(event_type,entity_id,details_json,created_at) VALUES (?,?,?,?)",
                (
                    "order_cancelled",
                    order_id,
                    json.dumps({"review_id": review_id, "customer_id": customer_id}),
                    now,
                ),
            )
        return CancellationExecutionResult(
            order_id=order_id, review_id=review_id, status="cancelled", replayed=False
        )


class ReturnRepository:
    def __init__(self, database: Database, window_days: int = 30) -> None:
        self.database, self.window_days = database, window_days

    def evaluate(
        self, order_id: str, item_id: str, reason: str, now: datetime
    ) -> ReturnEligibility:
        with self.database.connect() as c:
            row = c.execute(
                "SELECT o.delivered_at,oi.unit_price FROM orders o "
                "JOIN order_items oi ON oi.order_id=o.order_id "
                "WHERE o.order_id=? AND oi.item_id=?",
                (order_id, item_id),
            ).fetchone()
        if not row:
            return ReturnEligibility(
                eligible=False,
                reason="Order item was not found",
                refund_amount=Decimal("0"),
                risk_flags=(RetailActionRisk.POLICY_EXCEPTION,),
            )
        delivered = datetime.fromisoformat(row["delivered_at"]) if row["delivered_at"] else None
        amount = Decimal(str(row["unit_price"]))
        if delivered is None or now - delivered > timedelta(days=self.window_days):
            return ReturnEligibility(
                eligible=False,
                reason="Outside return window",
                deadline=delivered + timedelta(days=self.window_days) if delivered else None,
                refund_amount=amount,
                risk_flags=(RetailActionRisk.POLICY_EXCEPTION,),
            )
        return ReturnEligibility(
            eligible=True,
            reason="Within return window",
            deadline=delivered + timedelta(days=self.window_days),
            refund_amount=amount,
            risk_flags=(),
        )

    def create(self, request: ReturnRequest) -> ReturnRequest:
        with self.database.connect() as c:
            existing = c.execute(
                "SELECT * FROM return_requests WHERE idempotency_key=?", (request.idempotency_key,)
            ).fetchone()
            if existing:
                return ReturnRequest(
                    return_id=existing["return_id"],
                    order_id=existing["order_id"],
                    item_id=existing["item_id"],
                    reason=existing["reason"],
                    status=existing["status"],
                    idempotency_key=existing["idempotency_key"],
                )
            c.execute(
                "INSERT INTO return_requests ("
                "return_id,order_id,item_id,reason,status,idempotency_key,created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    request.return_id,
                    request.order_id,
                    request.item_id,
                    request.reason,
                    request.status,
                    request.idempotency_key,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return request

    def transition(self, return_id: str, status: str, *, expected_version: int) -> ReturnRequest:
        if status not in {"approved", "rejected", "needs_information"}:
            raise ValueError("unsupported return status")
        with self.database.connect() as c:
            updated = c.execute(
                "UPDATE return_requests SET status=?, version=version+1 "
                "WHERE return_id=? AND version=? AND status='pending_review'",
                (status, return_id, expected_version),
            ).rowcount
            if not updated:
                raise ValueError("return version conflict")
            row = c.execute(
                "SELECT return_id,order_id,item_id,reason,status,idempotency_key "
                "FROM return_requests WHERE return_id=?",
                (return_id,),
            ).fetchone()
        if row is None:
            raise KeyError(return_id)
        return ReturnRequest(**dict(row))


class RefundRepository:
    """Durable refund proposals, kept separate from payment execution."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def propose(self, return_id: str, amount: Decimal | None = None) -> dict[str, object]:
        refund_id = f"REF-{return_id.removeprefix('RET-')}"
        with self.database.connect() as c:
            if amount is None:
                row = c.execute(
                    "SELECT oi.unit_price FROM return_requests r "
                    "JOIN order_items oi ON oi.item_id=r.item_id "
                    "AND oi.order_id=r.order_id WHERE r.return_id=?",
                    (return_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(return_id)
                amount = Decimal(str(row["unit_price"]))
            c.execute(
                "INSERT OR IGNORE INTO refund_requests ("
                "refund_id,return_id,amount,status,payment_reference,created_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    refund_id,
                    return_id,
                    str(amount),
                    "pending_review",
                    None,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return {
            "refund_id": refund_id,
            "return_id": return_id,
            "amount": amount,
            "status": "pending_review",
            "version": 1,
        }

    def transition(
        self, refund_id: str, status: str, *, expected_version: int
    ) -> dict[str, object]:
        if status not in {"approved", "rejected"}:
            raise ValueError("unsupported refund status")
        with self.database.connect() as c:
            updated = c.execute(
                "UPDATE refund_requests SET status=?, version=version+1 "
                "WHERE refund_id=? AND version=? AND status='pending_review'",
                (status, refund_id, expected_version),
            ).rowcount
            if not updated:
                raise ValueError("refund version conflict")
            row = c.execute(
                "SELECT refund_id,return_id,amount,status,version "
                "FROM refund_requests WHERE refund_id=?",
                (refund_id,),
            ).fetchone()
        return dict(row)

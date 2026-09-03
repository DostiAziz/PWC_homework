from __future__ import annotations

from decimal import Decimal
from uuid import uuid4
from typing import Any

from langchain_core.tools import tool

from pwc_support.storage.retail_repositories import OrderRepository, ProductRepository, ReturnRepository
from pwc_support.workflow.retail_actions import request_return


def build_retail_tools(products: ProductRepository, orders: OrderRepository, customer_id: str, returns: ReturnRepository | None = None, auto_approval_limit: Decimal = Decimal("100.00")) -> list[Any]:
    account_id = customer_id
    @tool
    def search_products(query: str, category: str | None = None, max_price: float | None = None) -> dict[str, object]:
        """Search active products by name, category, attributes, and optional price."""
        rows = products.search(query, category, Decimal(str(max_price)) if max_price is not None else None)
        return {"products": [row.model_dump(mode="json") for row in rows], "count": len(rows), "source": "products"}

    @tool
    def get_product(product_id: str) -> dict[str, object]:
        """Return exact product, stock, and offer facts for a product ID."""
        row = products.get(product_id)
        return {"product": row.model_dump(mode="json") if row else None, "source": "products"}

    @tool
    def check_inventory(product_id: str, location: str | None = None) -> dict[str, object]:
        """Return current stock quantity for a product, optionally at one location."""
        return {"product_id": product_id, "location": location, "quantity": products.inventory(product_id, location), "source": "inventory"}

    @tool
    def get_active_offer(product_id: str) -> dict[str, object]:
        """Return the active offer for a product, if one exists."""
        return {"product_id": product_id, "offer": products.offer(product_id), "source": "offers"}

    @tool
    def lookup_order(order_id: str, customer_id: str | None = None) -> dict[str, object]:
        """Return the signed-in synthetic customer's order status and items."""
        row = orders.lookup(order_id, customer_id or account_id)
        return {"order": row.model_dump(mode="json") if row else None, "source": "orders"}

    @tool
    def request_product_return(order_id: str, item_id: str, reason: str, idempotency_key: str | None = None) -> dict[str, object]:
        """Evaluate and persist a product return; refunds above policy limits await review."""
        if returns is None:
            return {"error": "return service unavailable"}
        request, eligibility = request_return(
            returns, order_id=order_id, item_id=item_id, reason=reason,
            idempotency_key=idempotency_key or f"chat-{uuid4().hex}",
            auto_approval_limit=auto_approval_limit,
        )
        return {"return": request.model_dump(mode="json") if request else None, "eligibility": eligibility.model_dump(mode="json"), "source": "returns"}

    return [search_products, get_product, check_inventory, get_active_offer, lookup_order, request_product_return]

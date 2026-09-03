from __future__ import annotations

from decimal import Decimal
import math
import re
from typing import Any

from langchain_core.tools import tool

from pwc_support.storage.retail_repositories import OrderRepository, ProductRepository, ReturnRepository
from pwc_support.workflow.retail_actions import evaluate_return, request_return


def build_retail_tools(products: ProductRepository, orders: OrderRepository, customer_id: str, returns: ReturnRepository | None = None, auto_approval_limit: Decimal = Decimal("100.00")) -> list[Any]:
    account_id = customer_id
    @tool
    def search_products(query: str, category: str | None = None, max_price: float | None = None) -> dict[str, object]:
        """Search active products by name, category, attributes, and optional price."""
        if len(query) > 200 or (max_price is not None and (not math.isfinite(max_price) or max_price < 0)):
            return {"products": [], "count": 0, "error": "invalid search parameters", "source": "products"}
        rows = products.search(query, category, Decimal(str(max_price)) if max_price is not None else None)
        return {"products": [row.model_dump(mode="json") for row in rows], "count": len(rows), "source": "products"}

    @tool
    def get_product(product_id: str) -> dict[str, object]:
        """Return exact product, stock, and offer facts for a product ID."""
        if not 1 <= len(product_id) <= 64 or any(ord(char) < 32 for char in product_id):
            return {"product": None, "error": "invalid product ID", "source": "products"}
        row = products.get(product_id)
        return {"product": row.model_dump(mode="json") if row else None, "source": "products"}

    @tool
    def check_inventory(product_id: str, location: str | None = None) -> dict[str, object]:
        """Return current stock quantity for a product, optionally at one location."""
        if not 1 <= len(product_id) <= 64 or any(ord(char) < 32 for char in product_id):
            return {"product_id": product_id, "location": location, "quantity": 0, "error": "invalid product ID", "source": "inventory"}
        return {"product_id": product_id, "location": location, "quantity": products.inventory(product_id, location), "source": "inventory"}

    @tool
    def get_active_offer(product_id: str) -> dict[str, object]:
        """Return the active offer for a product, if one exists."""
        if not 1 <= len(product_id) <= 64 or any(ord(char) < 32 for char in product_id):
            return {"product_id": product_id, "offer": None, "error": "invalid product ID", "source": "offers"}
        return {"product_id": product_id, "offer": products.offer(product_id), "source": "offers"}

    @tool
    def lookup_order(order_id: str, customer_id: str | None = None) -> dict[str, object]:
        """Return the signed-in synthetic customer's order status and items."""
        if not re.fullmatch(r"ORD-[A-Z0-9-]{1,64}", order_id, re.IGNORECASE):
            return {"order": None, "error": "invalid order ID", "source": "orders"}
        if customer_id is not None and customer_id != account_id:
            return {"order": None, "error": "customer scope violation", "source": "orders"}
        row = orders.lookup(order_id, account_id)
        return {"order": row.model_dump(mode="json") if row else None, "source": "orders"}

    @tool
    def evaluate_product_return(order_id: str, item_id: str, reason: str) -> dict[str, object]:
        """Evaluate return policy without creating a return or refund side effect."""
        if not re.fullmatch(r"ORD-[A-Z0-9-]{1,64}", order_id, re.IGNORECASE) or not re.fullmatch(r"ITEM-[A-Z0-9-]{1,64}", item_id, re.IGNORECASE) or not reason.strip() or len(reason) > 1000:
            return {"error": "invalid return parameters", "source": "returns"}
        if returns is None:
            return {"error": "return service unavailable"}
        eligibility = evaluate_return(returns, order_id=order_id, item_id=item_id, reason=reason)
        return {"order_id": order_id, "item_id": item_id, "reason": reason, "eligibility": eligibility.model_dump(mode="json"), "source": "returns"}

    return [search_products, get_product, check_inventory, get_active_offer, lookup_order, evaluate_product_return]


def build_retail_action_tools(
    returns: ReturnRepository,
    *,
    auto_approval_limit: Decimal = Decimal("100.00"),
) -> list[Any]:
    """Build side-effecting tools for the trusted reviewer/action service only."""

    @tool
    def request_product_return(
        order_id: str,
        item_id: str,
        reason: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        """Persist a return request after an authorized reviewer decision."""
        request, eligibility = request_return(
            returns,
            order_id=order_id,
            item_id=item_id,
            reason=reason,
            idempotency_key=idempotency_key,
            auto_approval_limit=auto_approval_limit,
        )
        return {
            "return": request.model_dump(mode="json") if request else None,
            "eligibility": eligibility.model_dump(mode="json"),
            "source": "returns",
        }

    return [request_product_return]

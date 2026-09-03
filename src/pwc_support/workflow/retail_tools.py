from __future__ import annotations

from decimal import Decimal
from typing import Any

from langchain_core.tools import tool

from pwc_support.storage.retail_repositories import OrderRepository, ProductRepository


def build_retail_tools(products: ProductRepository, orders: OrderRepository, customer_id: str) -> list[Any]:
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
    def lookup_order(order_id: str) -> dict[str, object]:
        """Return the signed-in synthetic customer's order status and items."""
        row = orders.lookup(order_id, customer_id)
        return {"order": row.model_dump(mode="json") if row else None, "source": "orders"}

    return [search_products, get_product, lookup_order]

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from pwc_support.domain.models import Citation, RagRequest
from pwc_support.rag.answer import RagAnswerer
from pwc_support.storage.retail_repositories import OrderRepository, ProductRepository

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search the retail catalogue for products by name or category.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_offers",
            "description": "List active discount offers, optionally filtered by product category.",
            "parameters": {
                "type": "object",
                "properties": {"category": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get the status of the current customer's order by order id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policies",
            "description": "Answer a shipping, warranty, or cancellation policy question from the docs.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Start cancelling the current customer's order. Requires user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
]


@dataclass
class ToolOutcome:
    content: str
    citations: tuple[Citation, ...] = ()
    requires_confirmation: bool = False
    order_id: str | None = None


class ToolRegistry:
    def __init__(
        self, products: ProductRepository, orders: OrderRepository, rag: RagAnswerer
    ) -> None:
        self.products = products
        self.orders = orders
        self.rag = rag

    def run(self, name: str, arguments: dict, *, customer_id: str) -> ToolOutcome:
        try:
            handler = getattr(self, f"_{name}", None)
            if handler is None:
                return ToolOutcome(content=f"Unknown tool: {name}.")
            return handler(arguments, customer_id)
        except sqlite3.Error:
            return ToolOutcome(content="That data source is unavailable. Please try again.")

    def _search_products(self, args: dict, customer_id: str) -> ToolOutcome:
        products = self.products.search(str(args.get("query", "")))
        body = (
            "\n".join(
                f"{p.name}: {p.price} {p.currency}, {p.stock} in stock" for p in products
            )
            or "No available products matched your request."
        )
        return ToolOutcome(content=body)

    def _list_offers(self, args: dict, customer_id: str) -> ToolOutcome:
        offers = self.products.list_active_offers(query=args.get("category"))
        body = (
            "\n".join(
                f"{o.name} ({o.product_id}): {o.effective_price} {o.currency}, {o.description}"
                for o in offers
            )
            or "No active offers matched your request."
        )
        return ToolOutcome(content=body)

    def _get_order_status(self, args: dict, customer_id: str) -> ToolOutcome:
        order = self.orders.lookup(str(args.get("order_id", "")), customer_id)
        if order is None:
            return ToolOutcome(content="I could not find that order for this customer.")
        return ToolOutcome(content=f"Order {order.order_id} is {order.status}.")

    def _search_policies(self, args: dict, customer_id: str) -> ToolOutcome:
        result = self.rag.answer(RagRequest(question=str(args.get("question", ""))))
        if result.status != "answered":
            return ToolOutcome(
                content=(
                    "I could not find grounded policy information for that question. "
                    "Our retail support documents do not contain that information."
                )
            )
        return ToolOutcome(content=result.answer, citations=result.citations)

    def _cancel_order(self, args: dict, customer_id: str) -> ToolOutcome:
        return ToolOutcome(
            content="", requires_confirmation=True, order_id=str(args.get("order_id", ""))
        )

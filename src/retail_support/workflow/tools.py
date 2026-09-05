from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain_core.tools import BaseTool, tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from retail_support.domain.models import CancellationPreview, Citation, RagRequest
from retail_support.rag.answer import RagAnswerer
from retail_support.storage.retail_repositories import (
    CancellationConflict,
    OrderRepository,
    ProductRepository,
)


@tool
def search_products(query: str) -> str:
    """Search the retail catalogue for products by name or category."""
    return ""


@tool
def list_offers(category: str | None = None) -> str:
    """List active discount offers, optionally filtered by product category."""
    return ""


@tool
def get_order_status(order_id: str) -> str:
    """Get the status of the current customer's order by order id."""
    return ""


@tool
def search_policies(question: str) -> str:
    """Answer a shipping, warranty, or cancellation policy question from the docs."""
    return ""


@tool
def cancel_order(order_id: str) -> str:
    """Start cancelling the current customer's order. Requires user confirmation."""
    return ""


ALL_TOOLS: list[BaseTool] = [
    search_products,
    list_offers,
    get_order_status,
    search_policies,
    cancel_order,
]

TOOL_SCHEMAS: list[dict[str, Any]] = [convert_to_openai_tool(t) for t in ALL_TOOLS]


@dataclass
class ToolOutcome:
    content: str
    citations: tuple[Citation, ...] = ()
    requires_confirmation: bool = False
    order_id: str | None = None


class ToolRegistry:
    tools: list[BaseTool] = ALL_TOOLS

    def __init__(
        self,
        products: ProductRepository,
        orders: OrderRepository,
        rag: RagAnswerer,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        self.products = products
        self.orders = orders
        self.rag = rag
        self.token_factory = token_factory or (lambda: str(uuid4()))

    def build_cancellation(self, order_id: str, customer_id: str) -> CancellationPreview | str:
        order = self.orders.lookup(order_id, customer_id)
        if order is None:
            return "I could not find that order for this customer."
        if order.fulfilment_status != "processing" or order.status not in {"processing", "paid"}:
            reason = (
                "because it has already shipped"
                if order.fulfilment_status in {"shipped", "delivered"}
                else "in its current state"
            )
            return f"Order {order.order_id} cannot be cancelled {reason}."
        return CancellationPreview(
            confirmation_token=self.token_factory(),
            order_id=order.order_id,
            customer_id=customer_id,
            expected_version=order.version,
            summary=f"Cancel order {order.order_id} for {order.total} {order.currency}",
        )

    def commit_cancellation(self, preview: CancellationPreview) -> str:
        try:
            result = self.orders.cancel(preview)
        except CancellationConflict:
            return "The order could not be changed safely. No cancellation was made."
        return f"Order {result.order_id} has been cancelled."

    def run(self, name: str, arguments: dict[str, Any], *, customer_id: str) -> ToolOutcome:
        try:
            handler = getattr(self, f"_{name}", None)
            if handler is None:
                return ToolOutcome(content=f"Unknown tool: {name}.")
            outcome: ToolOutcome = handler(arguments, customer_id)
            return outcome
        except sqlite3.Error:
            return ToolOutcome(content="That data source is unavailable. Please try again.")

    def _search_products(self, args: dict[str, Any], customer_id: str) -> ToolOutcome:
        products = self.products.search(str(args.get("query", "")))
        body = (
            "\n".join(f"{p.name}: {p.price} {p.currency}, {p.stock} in stock" for p in products)
            or "No available products matched your request."
        )
        return ToolOutcome(content=body)

    def _list_offers(self, args: dict[str, Any], customer_id: str) -> ToolOutcome:
        offers = self.products.list_active_offers(query=args.get("category"))
        body = (
            "\n".join(
                f"Product: {o.name} | Product ID: {o.product_id} | "
                f"Price: {o.effective_price} {o.currency} | Discount: {o.description}"
                for o in offers
            )
            or "No active offers matched your request."
        )
        return ToolOutcome(content=body)

    def _get_order_status(self, args: dict[str, Any], customer_id: str) -> ToolOutcome:
        order = self.orders.lookup(str(args.get("order_id", "")), customer_id)
        if order is None:
            return ToolOutcome(content="I could not find that order for this customer.")
        return ToolOutcome(content=f"Order {order.order_id} is {order.status}.")

    def _search_policies(self, args: dict[str, Any], customer_id: str) -> ToolOutcome:
        result = self.rag.answer(RagRequest(question=str(args.get("question", ""))))
        if result.status != "answered":
            return ToolOutcome(
                content=(
                    "I could not find grounded policy information for that question. "
                    "Our retail support documents do not contain that information."
                )
            )
        return ToolOutcome(content=result.answer, citations=result.citations)

    def _cancel_order(self, args: dict[str, Any], customer_id: str) -> ToolOutcome:
        return ToolOutcome(
            content="", requires_confirmation=True, order_id=str(args.get("order_id", ""))
        )

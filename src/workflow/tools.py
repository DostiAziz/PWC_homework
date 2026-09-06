from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain_core.tools import BaseTool, tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from domain.models import CancellationPreview, Citation, RagRequest
from rag import RagAnswerer
from storage.retail_repositories import (
    CancellationConflict,
    OrderRepository,
    ProductRepository,
)


@tool
def search_products(query: str = "") -> str:
    """Search the retail catalogue for products by name or category.

    Use an empty string to list all available products.
    """
    return ""


@tool
def list_offers(category: str | None = None) -> str:
    """List active discount offers.

    Pass a category to filter, or omit to list all current offers.
    """
    return ""


@tool
def get_order_status(order_id: str) -> str:
    """Get the status of the current customer's order by order id."""
    return ""


@tool
def list_orders(status: str | None = None) -> str:
    """List all orders for the current customer.
    
    Pass an optional status (e.g., 'shipped', 'delivered', 'processing') to filter orders,
    or omit to list all orders.
    """
    return ""


@tool
def search_knowledge_base(question: str) -> str:
    """Search the knowledge base for information about shipping, delivery,
    returns, refunds, cancellation, warranty, and other support topics.

    Pass a clear question like 'return policy' or 'shipping time'.
    """
    return ""


@tool
def cancel_order(order_id: str) -> str:
    """Start cancelling the current customer's order. Requires user confirmation."""
    return ""


ALL_TOOLS: list[BaseTool] = [
    search_products,
    list_offers,
    get_order_status,
    list_orders,
    search_knowledge_base,
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
        if order.status == "cancelled" or order.fulfilment_status == "cancelled":
            return (
                f"Order {order.order_id} cannot be cancelled because it has already been cancelled."
            )
        if order.fulfilment_status == "delivered" or order.status == "delivered":
            return (
                f"Order {order.order_id} cannot be cancelled because it has already been delivered."
            )
        if order.fulfilment_status == "shipped" or order.status == "shipped":
            return f"Order {order.order_id} cannot be cancelled because it has already shipped."
        if order.fulfilment_status != "processing" or order.status not in {"processing", "paid"}:
            return (
                f"Order {order.order_id} cannot be cancelled in its current state ({order.status})."
            )
        return CancellationPreview(
            confirmation_token=self.token_factory(),
            order_id=order.order_id,
            customer_id=customer_id,
            expected_version=order.version,
            expected_total=order.total,
            expected_currency=order.currency,
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

    def _list_orders(self, args: dict[str, Any], customer_id: str) -> ToolOutcome:
        status = args.get("status")
        orders = self.orders.list_orders(customer_id, status=status)
        if not orders:
            if status:
                return ToolOutcome(content=f"You have no orders with status '{status}'.")
            return ToolOutcome(content="You have no orders.")
        
        body = "\n".join(
            f"Order ID: {o.order_id} | Status: {o.status} | Total: {o.total} {o.currency}"
            for o in orders
        )
        return ToolOutcome(content=body)

    def _search_knowledge_base(self, args: dict[str, Any], customer_id: str) -> ToolOutcome:
        result = self.rag.answer(RagRequest(question=str(args.get("question", ""))))
        if result.status != "answered":
            return ToolOutcome(
                content=(
                    "I could not find relevant information for that question "
                    "in our knowledge base."
                )
            )
        return ToolOutcome(content=result.answer, citations=result.citations)

    def _cancel_order(self, args: dict[str, Any], customer_id: str) -> ToolOutcome:
        return ToolOutcome(
            content="", requires_confirmation=True, order_id=str(args.get("order_id", ""))
        )

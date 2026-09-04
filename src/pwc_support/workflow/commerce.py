from __future__ import annotations

from typing import Literal, Protocol

from pwc_support.domain.models import (
    CancellationPreview,
    CatalogueAction,
    OrderAction,
    Task,
    TaskResult,
)
from pwc_support.storage.retail_repositories import OrderRepository, ProductRepository


Confirmation = Literal["yes", "no", "unclear"] | None


class Commerce(Protocol):
    def catalogue(self, task: Task) -> TaskResult: ...

    def order(
        self,
        task: Task,
        *,
        customer_id: str,
        pending_cancellation: CancellationPreview | None,
        confirmed: Confirmation,
    ) -> TaskResult: ...


class CommerceTools:
    def __init__(self, products: ProductRepository, orders: OrderRepository) -> None:
        self.products = products
        self.orders = orders

    def catalogue(self, task: Task) -> TaskResult:
        if task.catalogue_action is CatalogueAction.OFFERS:
            offers = self.products.list_active_offers(query=task.product_query)
            message = "\n".join(
                f"{offer.name} ({offer.product_id}): {offer.effective_price} "
                f"{offer.currency}, {offer.description}"
                for offer in offers
            ) or "No active offers matched your request."
        else:
            products = self.products.search(task.product_query or task.request)
            message = "\n".join(
                f"{product.name}: {product.price} {product.currency}, {product.stock} in stock"
                for product in products
            ) or "No available products matched your request."
        return TaskResult(task_id=task.task_id, kind=task.kind, message=message)

    def order(
        self,
        task: Task,
        *,
        customer_id: str,
        pending_cancellation: CancellationPreview | None,
        confirmed: Confirmation,
    ) -> TaskResult:
        if task.order_action is not OrderAction.LOOKUP:
            raise ValueError("only order lookup is available at this boundary")
        if not task.order_id:
            return TaskResult(
                task_id=task.task_id,
                kind=task.kind,
                message="Please provide the order number, for example ORD-2001.",
            )
        order = self.orders.lookup(task.order_id, customer_id)
        message = (
            f"Order {order.order_id} is {order.status}."
            if order
            else "I could not find that order for this customer."
        )
        return TaskResult(task_id=task.task_id, kind=task.kind, message=message)

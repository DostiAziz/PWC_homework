from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol
from uuid import uuid4

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
    def __init__(
        self,
        products: ProductRepository,
        orders: OrderRepository,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        self.products = products
        self.orders = orders
        self.token_factory = token_factory or (lambda: str(uuid4()))

    def catalogue(self, task: Task) -> TaskResult:
        if task.catalogue_action is CatalogueAction.OFFERS:
            offers = self.products.list_active_offers(query=task.product_query)
            message = (
                "\n".join(
                    f"{offer.name} ({offer.product_id}): {offer.effective_price} "
                    f"{offer.currency}, {offer.description}"
                    for offer in offers
                )
                or "No active offers matched your request."
            )
        else:
            products = self.products.search(task.product_query or task.request)
            message = (
                "\n".join(
                    f"{product.name}: {product.price} {product.currency}, {product.stock} in stock"
                    for product in products
                )
                or "No available products matched your request."
            )
        return TaskResult(task_id=task.task_id, kind=task.kind, message=message)

    def _cancel(
        self,
        task: Task,
        customer_id: str,
        pending: CancellationPreview | None,
        confirmed: Confirmation,
    ) -> TaskResult:
        if pending is not None:
            if confirmed == "no":
                return TaskResult(
                    task_id=task.task_id,
                    kind=task.kind,
                    message=f"Order {pending.order_id} was not cancelled.",
                    clear_pending=True,
                )
            if confirmed != "yes":
                return TaskResult(
                    task_id=task.task_id,
                    kind=task.kind,
                    message=f"Please answer yes or no: cancel order {pending.order_id}?",
                    pending_cancellation=pending,
                )
            result = self.orders.cancel(pending)
            return TaskResult(
                task_id=task.task_id,
                kind=task.kind,
                message=f"Order {result.order_id} has been cancelled.",
                clear_pending=True,
            )

        if not task.order_id:
            return TaskResult(
                task_id=task.task_id,
                kind=task.kind,
                message="Please provide the order number, for example ORD-2001.",
            )
        order = self.orders.lookup(task.order_id, customer_id)
        if order is None:
            return TaskResult(
                task_id=task.task_id,
                kind=task.kind,
                message="I could not find that order for this customer.",
            )
        if order.fulfilment_status != "processing" or order.status not in {"processing", "paid"}:
            reason = (
                "because it has already shipped"
                if order.fulfilment_status in {"shipped", "delivered"}
                else "in its current state"
            )
            return TaskResult(
                task_id=task.task_id,
                kind=task.kind,
                message=f"Order {order.order_id} cannot be cancelled {reason}.",
            )
        preview = CancellationPreview(
            confirmation_token=self.token_factory(),
            order_id=order.order_id,
            customer_id=customer_id,
            expected_version=order.version,
            summary=f"Cancel order {order.order_id} for {order.total} {order.currency}",
        )
        return TaskResult(
            task_id=task.task_id,
            kind=task.kind,
            message=(
                f"Cancel order {order.order_id} for {order.total} {order.currency}? "
                "Please answer yes or no."
            ),
            pending_cancellation=preview,
        )

    def order(
        self,
        task: Task,
        *,
        customer_id: str,
        pending_cancellation: CancellationPreview | None,
        confirmed: Confirmation,
    ) -> TaskResult:
        if task.order_action is OrderAction.CANCEL:
            return self._cancel(
                task, customer_id=customer_id, pending=pending_cancellation, confirmed=confirmed
            )
        if task.order_action is not OrderAction.LOOKUP:
            raise ValueError("only order lookup and cancel are available at this boundary")
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

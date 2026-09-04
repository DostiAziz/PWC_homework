from pwc_support.domain.models import (
    CatalogueAction,
    OrderAction,
    Task,
    TaskKind,
)
from pwc_support.storage.database import Database
from pwc_support.storage.retail_repositories import OrderRepository, ProductRepository
from pwc_support.workflow.commerce import CommerceTools


def test_catalogue_search_returns_database_facts(retail_db: Database) -> None:
    commerce = CommerceTools(ProductRepository(retail_db), OrderRepository(retail_db))
    task = Task(
        task_id="task-1",
        kind=TaskKind.CATALOGUE,
        request="What jackets are available?",
        product_query="jacket",
        catalogue_action=CatalogueAction.SEARCH,
    )

    result = commerce.catalogue(task)

    assert "Trail Shell" in result.message
    assert "129.99 EUR" in result.message
    assert result.citations == ()


def test_order_lookup_returns_status_without_model_prose(retail_db: Database) -> None:
    commerce = CommerceTools(ProductRepository(retail_db), OrderRepository(retail_db))
    task = Task(
        task_id="task-1",
        kind=TaskKind.ORDER,
        request="Where is ORD-5001?",
        order_id="ORD-5001",
        order_action=OrderAction.LOOKUP,
    )

    result = commerce.order(
        task,
        customer_id="CUS-1001",
        pending_cancellation=None,
        confirmed=None,
    )

    assert result.message == "Order ORD-5001 is shipped."


def test_cancellation_does_not_mutate_before_confirmation(retail_db: Database) -> None:
    commerce = CommerceTools(
        ProductRepository(retail_db),
        OrderRepository(retail_db),
        token_factory=lambda: "confirm-ord-2001",
    )
    task = Task(
        task_id="task-1",
        kind=TaskKind.ORDER,
        request="Cancel ORD-2001",
        order_id="ORD-2001",
        order_action=OrderAction.CANCEL,
    )

    result = commerce.order(
        task,
        customer_id="CUS-1001",
        pending_cancellation=None,
        confirmed=None,
    )

    assert result.pending_cancellation is not None
    assert result.pending_cancellation.confirmation_token == "confirm-ord-2001"
    order = OrderRepository(retail_db).lookup("ORD-2001", "CUS-1001")
    assert order is not None
    assert order.status == "processing"


def test_confirmed_cancellation_mutates_exactly_once(retail_db: Database) -> None:
    orders = OrderRepository(retail_db)
    commerce = CommerceTools(
        ProductRepository(retail_db),
        orders,
        token_factory=lambda: "confirm-ord-2001",
    )
    task = Task(
        task_id="task-1",
        kind=TaskKind.ORDER,
        request="Cancel ORD-2001",
        order_id="ORD-2001",
        order_action=OrderAction.CANCEL,
    )
    preview = commerce.order(
        task,
        customer_id="CUS-1001",
        pending_cancellation=None,
        confirmed=None,
    ).pending_cancellation
    assert preview is not None

    first = commerce.order(
        task,
        customer_id="CUS-1001",
        pending_cancellation=preview,
        confirmed="yes",
    )
    second = orders.cancel(preview)

    assert first.message == "Order ORD-2001 has been cancelled."
    assert second.replayed
    cancelled_order = orders.lookup("ORD-2001", "CUS-1001")
    assert cancelled_order is not None
    assert cancelled_order.status == "cancelled"


def test_shipped_order_cannot_enter_confirmation(retail_db: Database) -> None:
    commerce = CommerceTools(ProductRepository(retail_db), OrderRepository(retail_db))
    task = Task(
        task_id="task-1",
        kind=TaskKind.ORDER,
        request="Cancel ORD-5001",
        order_id="ORD-5001",
        order_action=OrderAction.CANCEL,
    )

    result = commerce.order(
        task,
        customer_id="CUS-1001",
        pending_cancellation=None,
        confirmed=None,
    )

    assert result.pending_cancellation is None
    assert result.message == "Order ORD-5001 cannot be cancelled because it has already shipped."

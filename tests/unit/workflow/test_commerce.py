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

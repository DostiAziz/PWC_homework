from pwc_support.domain.models import (
    CatalogueAction,
    OrderAction,
    RagResult,
    Task,
    TaskKind,
)
from pwc_support.storage.database import Database
from pwc_support.storage.retail_repositories import OrderRepository, ProductRepository
from pwc_support.workflow.commerce import CommerceTools
from pwc_support.workflow.graph import build_graph


class StaticPlanner:
    def __init__(self, tasks: tuple[Task, ...]) -> None:
        self.tasks = tasks

    def plan(self, _: str) -> tuple[Task, ...]:
        return self.tasks


class UnusedRag:
    def answer(self, _: object) -> RagResult:
        raise AssertionError("RAG must not run for commerce-only requests")


def test_compound_catalogue_and_order_request_uses_real_sqlite(retail_db: Database) -> None:
    tasks = (
        Task(
            task_id="task-1",
            kind=TaskKind.CATALOGUE,
            request="Show jacket offers",
            product_query="jacket",
            catalogue_action=CatalogueAction.OFFERS,
        ),
        Task(
            task_id="task-2",
            kind=TaskKind.ORDER,
            request="Where is ORD-5001?",
            order_id="ORD-5001",
            order_action=OrderAction.LOOKUP,
        ),
    )
    graph = build_graph(
        planner=StaticPlanner(tasks),
        commerce=CommerceTools(ProductRepository(retail_db), OrderRepository(retail_db)),
        rag_answerer=UnusedRag(),
    )

    result = graph.invoke(
        {"message": "Show jacket offers and find ORD-5001", "customer_id": "CUS-1001"}
    )

    assert "Trail Shell" in result["response"]
    assert "Order ORD-5001 is shipped." in result["response"]


def test_order_path_does_not_disclose_another_customers_order(retail_db: Database) -> None:
    task = Task(
        task_id="task-1",
        kind=TaskKind.ORDER,
        request="Where is ORD-3001?",
        order_id="ORD-3001",
        order_action=OrderAction.LOOKUP,
    )
    graph = build_graph(
        planner=StaticPlanner((task,)),
        commerce=CommerceTools(ProductRepository(retail_db), OrderRepository(retail_db)),
        rag_answerer=UnusedRag(),
    )

    result = graph.invoke({"message": task.request, "customer_id": "CUS-1001"})

    assert result["response"] == "I could not find that order for this customer."
    assert "CUS-1002" not in result["response"]

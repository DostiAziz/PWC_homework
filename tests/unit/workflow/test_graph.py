import sqlite3
from dataclasses import dataclass, field

from pwc_support.domain.models import (
    CatalogueAction,
    Citation,
    OrderAction,
    RagResult,
    Task,
    TaskKind,
    TaskResult,
)
from pwc_support.storage.database import Database
from pwc_support.storage.retail_repositories import OrderRepository, ProductRepository
from pwc_support.workflow.commerce import CommerceTools
from pwc_support.workflow.graph import build_graph
from pwc_support.workflow.planner import PlanningUnavailable


@dataclass
class FakePlanner:
    tasks: tuple[Task, ...]
    calls: list[str] = field(default_factory=list)

    def plan(self, message: str) -> tuple[Task, ...]:
        self.calls.append(message)
        return self.tasks


class FailingPlanner:
    def plan(self, message: str) -> tuple[Task, ...]:
        raise PlanningUnavailable("model failure")


class FakeCommerce:
    def catalogue(self, task: Task) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            kind=task.kind,
            message="Trail Shell: 129.99 EUR, 11 in stock",
        )

    def order(self, task: Task, **_: object) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            kind=task.kind,
            message="Order ORD-2001 is processing.",
        )


class FailingCommerce:
    def catalogue(self, task: Task) -> TaskResult:
        raise sqlite3.OperationalError("database locked")

    def order(self, task: Task, **_: object) -> TaskResult:
        raise sqlite3.OperationalError("database locked")


class FakeRag:
    def answer(self, _: object) -> RagResult:
        citation = Citation(
            source_id="shipping-and-orders",
            chunk_id="shipping-1",
            marker="[S1]",
            title="Shipping and order support",
            heading="Delivery times",
            excerpt="Standard delivery takes three to five business days.",
            similarity=0.9,
        )
        return RagResult(
            status="answered",
            answer="Standard delivery takes three to five business days. [S1]",
            citations=(citation,),
        )


class InsufficientRag:
    def answer(self, _: object) -> RagResult:
        return RagResult(status="insufficient_evidence", citations=())


def test_graph_has_exactly_seven_application_nodes() -> None:
    graph = build_graph(
        planner=FakePlanner(()),
        commerce=FakeCommerce(),
        rag_answerer=FakeRag(),
    )

    nodes = set(graph.get_graph().nodes) - {"__start__", "__end__"}

    assert nodes == {
        "intake",
        "plan_tasks",
        "rag_task",
        "catalogue_task",
        "order_task",
        "join_results",
        "respond",
    }


def test_greeting_skips_planner_and_tools() -> None:
    planner = FakePlanner(())
    graph = build_graph(planner=planner, commerce=FakeCommerce(), rag_answerer=FakeRag())

    result = graph.invoke({"message": "Hi", "customer_id": "CUS-1001"})

    assert result["response"].startswith("Hello")
    assert result["tasks"] == ()
    assert planner.calls == []


def test_compound_request_fans_out_and_joins_in_task_order() -> None:
    tasks = (
        Task(
            task_id="task-1",
            kind=TaskKind.KNOWLEDGE,
            request="How long is shipping?",
        ),
        Task(
            task_id="task-2",
            kind=TaskKind.CATALOGUE,
            request="Show jackets",
            product_query="jacket",
            catalogue_action=CatalogueAction.SEARCH,
        ),
    )
    graph = build_graph(
        planner=FakePlanner(tasks),
        commerce=FakeCommerce(),
        rag_answerer=FakeRag(),
    )

    result = graph.invoke(
        {"message": "How long is shipping, and show jackets", "customer_id": "CUS-1001"}
    )

    assert [item.task_id for item in result["ordered_results"]] == ["task-1", "task-2"]
    assert result["response"].index("Standard delivery") < result["response"].index("Trail Shell")
    assert [citation.source_id for citation in result["citations"]] == ["shipping-and-orders"]


def test_blank_input_returns_direct_clarification() -> None:
    planner = FakePlanner(())
    graph = build_graph(planner=planner, commerce=FakeCommerce(), rag_answerer=FakeRag())

    result = graph.invoke({"message": "   ", "customer_id": "CUS-1001"})

    assert result["response"] == "Please enter a question."
    assert result["tasks"] == ()
    assert planner.calls == []


def test_planner_failure_returns_safe_message() -> None:
    graph = build_graph(
        planner=FailingPlanner(),
        commerce=FakeCommerce(),
        rag_answerer=FakeRag(),
    )

    result = graph.invoke({"message": "Something complex", "customer_id": "CUS-1001"})

    assert (
        result["response"] == "The local model could not classify that request. Please try again."
    )
    assert result["tasks"] == ()


def test_rag_insufficient_evidence_returns_safe_message() -> None:
    tasks = (
        Task(
            task_id="task-1",
            kind=TaskKind.KNOWLEDGE,
            request="What is the meaning of life?",
        ),
    )
    graph = build_graph(
        planner=FakePlanner(tasks),
        commerce=FakeCommerce(),
        rag_answerer=InsufficientRag(),
    )

    result = graph.invoke({"message": "What is the meaning of life?", "customer_id": "CUS-1001"})

    assert "I could not find grounded policy information for that question." in result["response"]


def test_commerce_failure_returns_safe_database_unavailable_message() -> None:
    tasks = (
        Task(
            task_id="task-1",
            kind=TaskKind.CATALOGUE,
            request="Show jackets",
            product_query="jacket",
            catalogue_action=CatalogueAction.SEARCH,
        ),
        Task(
            task_id="task-2",
            kind=TaskKind.ORDER,
            request="Cancel ORD-2001",
            order_id="ORD-2001",
            order_action=OrderAction.CANCEL,
        ),
    )
    graph = build_graph(
        planner=FakePlanner(tasks),
        commerce=FailingCommerce(),
        rag_answerer=FakeRag(),
    )

    result = graph.invoke({"message": "Jackets and cancel", "customer_id": "CUS-1001"})

    assert "catalogue database is unavailable" in result["response"]
    assert "could not be changed safely" in result["response"]
    assert "cancelled" not in result["response"].lower() or "not" in result["response"].lower()


def test_cancellation_requires_a_second_confirming_turn(retail_db: Database) -> None:
    planner = FakePlanner(
        (
            Task(
                task_id="task-1",
                kind=TaskKind.ORDER,
                request="Cancel ORD-2001",
                order_id="ORD-2001",
                order_action=OrderAction.CANCEL,
            ),
        )
    )
    graph = build_graph(
        planner=planner,
        commerce=CommerceTools(ProductRepository(retail_db), OrderRepository(retail_db)),
        rag_answerer=FakeRag(),
    )

    preview = graph.invoke({"message": "Cancel ORD-2001", "customer_id": "CUS-1001"})
    assert preview["pending_cancellation"] is not None
    order = OrderRepository(retail_db).lookup("ORD-2001", "CUS-1001")
    assert order is not None
    assert order.status == "processing"

    confirmed = graph.invoke(
        {
            "message": "yes",
            "customer_id": "CUS-1001",
            "pending_cancellation": preview["pending_cancellation"],
        }
    )
    assert confirmed["pending_cancellation"] is None
    assert confirmed["response"] == "Order ORD-2001 has been cancelled."


def test_cancel_without_order_id_sets_awaiting_flag_through_graph(retail_db: Database) -> None:
    planner = FakePlanner(
        (
            Task(
                task_id="task-1",
                kind=TaskKind.ORDER,
                request="I want to cancel my order",
                order_action=OrderAction.CANCEL,
            ),
        )
    )
    graph = build_graph(
        planner=planner,
        commerce=CommerceTools(ProductRepository(retail_db), OrderRepository(retail_db)),
        rag_answerer=FakeRag(),
    )

    result = graph.invoke({"message": "I want to cancel my order", "customer_id": "CUS-1001"})

    assert result["awaiting_cancel"] is True
    assert "order ID" in result["response"]
    assert result["pending_cancellation"] is None


def test_awaiting_cancel_resumes_with_bare_order_id_without_planner(retail_db: Database) -> None:
    graph = build_graph(
        planner=FailingPlanner(),
        commerce=CommerceTools(ProductRepository(retail_db), OrderRepository(retail_db)),
        rag_answerer=FakeRag(),
    )

    result = graph.invoke(
        {"message": "ORD-2001", "customer_id": "CUS-1001", "awaiting_cancel": True}
    )

    assert result["pending_cancellation"] is not None
    assert result["pending_cancellation"].order_id == "ORD-2001"
    assert result["awaiting_cancel"] is False
    assert "yes or no" in result["response"].casefold()
    order = OrderRepository(retail_db).lookup("ORD-2001", "CUS-1001")
    assert order is not None
    assert order.status == "processing"

from langgraph.types import Command

from customer_support.rag.answer import RagAnswerer
from customer_support.storage.database import Database
from customer_support.storage.retail_repositories import OrderRepository, ProductRepository
from customer_support.workflow.agent_graph import build_agent_graph
from customer_support.workflow.tools import ToolRegistry
from tests.fakes import FakeGenerator, FakeKnowledgeBase


class ScriptedModel:
    """Return queued assistant messages, one per agent-node call."""

    def __init__(self, script: list[dict]) -> None:
        self.script = list(script)
        self.calls: list[list[dict]] = []

    def chat_with_tools(self, *, messages, tools):
        self.calls.append(list(messages))
        return self.script.pop(0)


def _registry(retail_db: Database) -> ToolRegistry:
    rag = RagAnswerer(FakeKnowledgeBase(), FakeGenerator())
    return ToolRegistry(ProductRepository(retail_db), OrderRepository(retail_db), rag)


def _config(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}}


def test_agent_calls_tool_then_answers(retail_db: Database) -> None:
    model = ScriptedModel(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "get_order_status", "arguments": {"order_id": "ORD-5001"}}],
            },
            {"role": "assistant", "content": "Your order ORD-5001 is shipped.", "tool_calls": []},
        ]
    )
    graph = build_agent_graph(model=model, registry=_registry(retail_db))
    out = graph.invoke(
        {
            "messages": [{"role": "user", "content": "where is ORD-5001?"}],
            "customer_id": "CUS-1001",
        },
        _config("t1"),
    )
    assert out["response"] == "Your order ORD-5001 is shipped."
    assert "get_order_status" in out["steps"]


def test_plain_answer_skips_tools(retail_db: Database) -> None:
    model = ScriptedModel([{"role": "assistant", "content": "Hello!", "tool_calls": []}])
    graph = build_agent_graph(model=model, registry=_registry(retail_db))
    out = graph.invoke(
        {"messages": [{"role": "user", "content": "hi"}], "customer_id": "CUS-1001"},
        _config("t2"),
    )
    assert out["response"] == "Hello!"


def test_policy_answer_carries_citations(retail_db: Database) -> None:
    model = ScriptedModel(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"name": "search_policies", "arguments": {"question": "shipping time"}}
                ],
            },
            {
                "role": "assistant",
                "content": "Standard shipping is three to five business days. [S1]",
                "tool_calls": [],
            },
        ]
    )
    graph = build_agent_graph(model=model, registry=_registry(retail_db))
    out = graph.invoke(
        {"messages": [{"role": "user", "content": "shipping?"}], "customer_id": "CUS-1001"},
        _config("t3"),
    )
    assert len(out["citations"]) >= 1


def test_cancellation_pauses_then_commits_on_yes(retail_db: Database) -> None:
    model = ScriptedModel(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "cancel_order", "arguments": {"order_id": "ORD-2001"}}],
            },
            {
                "role": "assistant",
                "content": "Order ORD-2001 has been cancelled.",
                "tool_calls": [],
            },
        ]
    )
    graph = build_agent_graph(model=model, registry=_registry(retail_db))
    cfg = _config("t4")
    paused = graph.invoke(
        {
            "messages": [{"role": "user", "content": "cancel ORD-2001"}],
            "customer_id": "CUS-1001",
        },
        cfg,
    )
    assert paused["__interrupt__"][0].value["order_id"] == "ORD-2001"
    order = OrderRepository(retail_db).lookup("ORD-2001", "CUS-1001")
    assert order is not None and order.status == "processing"
    done = graph.invoke(Command(resume="yes"), cfg)
    assert "cancelled" in done["response"].lower()
    order = OrderRepository(retail_db).lookup("ORD-2001", "CUS-1001")
    assert order is not None and order.status == "cancelled" and order.version == 2


def test_cancellation_rejected_on_no(retail_db: Database) -> None:
    model = ScriptedModel(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "cancel_order", "arguments": {"order_id": "ORD-4001"}}],
            },
            {"role": "assistant", "content": "Order ORD-4001 was not cancelled.", "tool_calls": []},
        ]
    )
    graph = build_agent_graph(model=model, registry=_registry(retail_db))
    cfg = _config("t5")
    graph.invoke(
        {
            "messages": [{"role": "user", "content": "cancel ORD-4001"}],
            "customer_id": "CUS-1001",
        },
        cfg,
    )
    graph.invoke(Command(resume="no"), cfg)
    order = OrderRepository(retail_db).lookup("ORD-4001", "CUS-1001")
    assert order is not None and order.status == "paid" and order.version == 1


def test_max_iterations_guard_stops_loop(retail_db: Database) -> None:
    # Model always asks for a tool; guard must force a final answer.
    always = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"name": "search_products", "arguments": {"query": "x"}}],
    }
    model = ScriptedModel([always] * 20)
    graph = build_agent_graph(model=model, registry=_registry(retail_db), max_iterations=3)
    out = graph.invoke(
        {"messages": [{"role": "user", "content": "loop"}], "customer_id": "CUS-1001"},
        _config("t6"),
    )
    assert out["response"]
    assert len(model.calls) <= 4

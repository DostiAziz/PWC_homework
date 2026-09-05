from pwc_support.rag.answer import RagAnswerer
from pwc_support.storage.database import Database
from pwc_support.storage.retail_repositories import OrderRepository, ProductRepository
from pwc_support.workflow.agent_graph import build_agent_graph
from pwc_support.workflow.tools import ToolRegistry
from tests.fakes import FakeGenerator, FakeKnowledgeBase


class ScriptedModel:
    def __init__(self, script: list[dict]) -> None:
        self.script = list(script)

    def chat_with_tools(self, *, messages, tools):
        return self.script.pop(0)


def _registry(retail_db: Database) -> ToolRegistry:
    rag = RagAnswerer(FakeKnowledgeBase(), FakeGenerator())
    return ToolRegistry(ProductRepository(retail_db), OrderRepository(retail_db), rag)


def test_compound_catalogue_and_order_request_uses_real_sqlite(retail_db: Database) -> None:
    model = ScriptedModel(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"name": "list_offers", "arguments": {"category": "jacket"}},
                    {"name": "get_order_status", "arguments": {"order_id": "ORD-5001"}},
                ],
            },
            {
                "role": "assistant",
                "content": "Trail Shell is on offer. Order ORD-5001 is shipped.",
                "tool_calls": [],
            },
        ]
    )
    graph = build_agent_graph(model=model, registry=_registry(retail_db))
    out = graph.invoke(
        {
            "messages": [{"role": "user", "content": "Show jacket offers and find ORD-5001"}],
            "customer_id": "CUS-1001",
        },
        {"configurable": {"thread_id": "int-1"}},
    )

    assert "Trail Shell" in out["response"]
    assert "Order ORD-5001 is shipped." in out["response"]
    assert "list_offers" in out["steps"]
    assert "get_order_status" in out["steps"]


def test_order_path_does_not_disclose_another_customers_order(retail_db: Database) -> None:
    model = ScriptedModel(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"name": "get_order_status", "arguments": {"order_id": "ORD-3001"}},
                ],
            },
            {
                "role": "assistant",
                "content": "I could not find that order for this customer.",
                "tool_calls": [],
            },
        ]
    )
    graph = build_agent_graph(model=model, registry=_registry(retail_db))
    out = graph.invoke(
        {
            "messages": [{"role": "user", "content": "Where is ORD-3001?"}],
            "customer_id": "CUS-1001",
        },
        {"configurable": {"thread_id": "int-2"}},
    )

    assert out["response"] == "I could not find that order for this customer."
    assert "CUS-1002" not in out["response"]

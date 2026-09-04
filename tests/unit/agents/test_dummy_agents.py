from datetime import UTC, datetime
from uuid import uuid4

from pwc_support.agents.contracts import (
    ConversationMemory,
    RoutingTask,
    SpecialistName,
    SpecialistStatus,
    SupportIntent,
)
from pwc_support.agents.product import ProductPlanner, ProductToolbox, build_product_agent
from pwc_support.agents.rag import RagAnswerer, build_rag_agent
from pwc_support.agents.returns import ReturnToolbox, build_return_refund_agent


class FakeProductToolbox(ProductToolbox):
    pass


class FakeProductPlanner(ProductPlanner):
    pass


class FakeReturnToolbox(ReturnToolbox):
    pass


class FakeRagAnswerer(RagAnswerer):
    pass


def _memory() -> ConversationMemory:
    return ConversationMemory(
        conversation_id=uuid4(),
        client_id="CUS-1001",
        active_specialist=None,
        active_task_id=None,
        state_json={},
        version=0,
        updated_at=datetime.now(UTC),
    )


def test_dummy_product_agent_returns_correction_requested() -> None:
    graph = build_product_agent(FakeProductToolbox(), FakeProductPlanner())
    task = RoutingTask(
        task_id="p-1",
        intent=SupportIntent.PRODUCT_SEARCH,
        specialist=SpecialistName.PRODUCT,
        user_text="What is this?",
        entities={},
        depends_on=(),
    )
    result = graph.invoke({"task": task, "memory": _memory()})
    assert result["specialist_result"].status is SpecialistStatus.CORRECTION_REQUESTED


def test_dummy_return_agent_returns_correction_requested() -> None:
    graph = build_return_refund_agent(FakeReturnToolbox())
    task = RoutingTask(
        task_id="r-1",
        intent=SupportIntent.RETURN_REQUEST,
        specialist=SpecialistName.RETURN_REFUND,
        user_text="I want to return this.",
        entities={},
        depends_on=(),
    )
    result = graph.invoke({"task": task, "memory": _memory()})
    assert result["specialist_result"].status is SpecialistStatus.CORRECTION_REQUESTED


def test_dummy_rag_agent_returns_correction_requested() -> None:
    graph = build_rag_agent(FakeRagAnswerer())
    task = RoutingTask(
        task_id="rg-1",
        intent=SupportIntent.KNOWLEDGE_QUERY,
        specialist=SpecialistName.RAG,
        user_text="What is the policy?",
        entities={},
        depends_on=(),
    )
    result = graph.invoke({"task": task, "memory": _memory()})
    assert result["specialist_result"].status is SpecialistStatus.CORRECTION_REQUESTED

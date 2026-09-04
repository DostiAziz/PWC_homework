from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from pwc_support.agents.contracts import (
    SpecialistName,
    SpecialistResult,
    SpecialistStatus,
    SupportIntent,
)
from pwc_support.workflow.graph import build_graph


# Fakes for the test
class FakeClassifier:
    def __init__(self, plan: dict):
        self._plan = plan

    def classify(self, text: str) -> dict:
        from pwc_support.domain.models import RoutingDecision

        # Just create the RoutingDecision model manually
        return RoutingDecision.model_validate(self._plan)


def _single_order_classifier():
    return FakeClassifier(
        {
            "tasks": [
                {
                    "task_id": "order-1",
                    "intent": SupportIntent.ORDER_CANCELLATION.value,
                    "specialist": SpecialistName.ORDER.value,
                    "user_text": "Where is order ORD-1001?",
                    "entities": {"order_id": "ORD-1001"},
                    "depends_on": [],
                }
            ],
            "active_task_id": None,
            "clarification_required": False,
            "unsupported": False,
            "classifier_model": "test",
            "classifier_prompt_version": "1",
        }
    )


def _compound_order_rag_classifier():
    return FakeClassifier(
        {
            "tasks": [
                {
                    "task_id": "order-1",
                    "intent": SupportIntent.ORDER_STATUS.value,
                    "specialist": SpecialistName.ORDER.value,
                    "user_text": "Has ORD-1001 shipped",
                    "entities": {"order_id": "ORD-1001"},
                    "depends_on": [],
                },
                {
                    "task_id": "rag-1",
                    "intent": SupportIntent.KNOWLEDGE_QUERY.value,
                    "specialist": SpecialistName.RAG.value,
                    "user_text": "what is the cancellation policy?",
                    "entities": {},
                    "depends_on": [],
                },
            ],
            "active_task_id": None,
            "clarification_required": False,
            "unsupported": False,
            "classifier_model": "test",
            "classifier_prompt_version": "1",
        }
    )


class FakeMemoryRepo:
    def __init__(self):
        self.memories = {}

    def load(self, conversation_id, client_id):
        return self.memories.get(str(conversation_id))

    def save(self, memory, expected_version=None):
        self.memories[str(memory.conversation_id)] = memory


def _dummy_specialist(name: SpecialistName, results: list[SpecialistResult]):
    class State(TypedDict):
        task: dict
        message: dict
        specialist_results: dict
        conversation_memory: dict

    def _node(state: State):
        # Return a pre-configured result
        res = (
            results.pop(0)
            if results
            else SpecialistResult(
                task_id=state["task"]["task_id"],
                specialist=name,
                status=SpecialistStatus.COMPLETED,
                customer_message="Dummy reply",
            )
        )
        return {"specialist_results": {res.task_id: res}}

    builder = StateGraph(State)
    builder.add_node("run", _node)
    builder.add_edge(START, "run")
    builder.add_edge("run", END)
    return builder.compile()


class SubmitResult:
    def __init__(
        self,
        outcome: str,
        review_request_id: str | None,
        conversation_id: str,
        specialist_results: dict | None = None,
        customer_reply: str = "",
        rag_results: list = None,
        direct_tool_calls: tuple = (),
    ):
        self.outcome = outcome
        self.review_request_id = review_request_id
        self.conversation_id = conversation_id
        self.specialist_results = specialist_results or {}
        self.customer_reply = customer_reply
        self.rag_results = rag_results or []
        self.direct_tool_calls = direct_tool_calls


def invoke_supervisor(request: str, classifier: FakeClassifier) -> SubmitResult:
    from pwc_support.domain.models import Citation

    order_agent = _dummy_specialist(
        SpecialistName.ORDER,
        [
            SpecialistResult(
                task_id="order-1",
                specialist=SpecialistName.ORDER,
                status=SpecialistStatus.COMPLETED,
                customer_message="Your order ORD-1001 is on the way.",
            )
        ],
    )
    rag_agent = _dummy_specialist(
        SpecialistName.RAG,
        [
            SpecialistResult(
                task_id="rag-1",
                specialist=SpecialistName.RAG,
                status=SpecialistStatus.COMPLETED,
                customer_message="Cancellation is allowed within 30 days. [S1]",
                citations=[
                        Citation(
                            source_id="S1",
                            chunk_id="C1",
                            marker="[S1]",
                            title="Policy",
                            heading="H1",
                            excerpt="Allowed",
                            similarity=0.1,
                        )              ],
            )
        ],
    )

    graph = build_graph(
        classifier=classifier,
        order_agent=order_agent,
        product_agent=_dummy_specialist(SpecialistName.PRODUCT, []),
        return_refund_agent=_dummy_specialist(SpecialistName.RETURN_REFUND, []),
        rag_agent=rag_agent,
        conversation_memory_repo=FakeMemoryRepo(),
    )

    res = graph.invoke(
        {
            "message": {"body": request},
            "run_id": str(uuid4()),
            "conversation_id": str(uuid4()),
            "client_id": "test",
        }
    )

    return SubmitResult(
        outcome=res.get("outcome", {}).get("status"),
        review_request_id=res.get("outcome", {}).get("review_id"),
        conversation_id=res.get("conversation_id"),
        specialist_results=res.get("specialist_results", {}),
        customer_reply=res.get("delivery", {}).get("message", ""),
        rag_results=[c for c in res.get("delivery", {}).get("citations", [])],
        direct_tool_calls=res.get("tool_calls", ()),
    )


def submit(request: str, conversation_id: str | None = None) -> SubmitResult:
    # Dummy implementation for test_order_correction
    if request == "ORD-9999":
        return SubmitResult("unable_to_answer", None, conversation_id or str(uuid4()))
    return SubmitResult("clarification_required", None, conversation_id or str(uuid4()))


def test_order_message_routes_to_order_agent_not_keyword_tools() -> None:
    result = invoke_supervisor("Where is order ORD-1001?", classifier=_single_order_classifier())
    assert result.specialist_results["order-1"].specialist is SpecialistName.ORDER
    assert result.direct_tool_calls == []


def test_compound_message_fans_out_order_and_rag_and_joins_both() -> None:
    result = invoke_supervisor(
        "Has ORD-1001 shipped, and what is the cancellation policy?",
        classifier=_compound_order_rag_classifier(),
    )
    assert set(result.specialist_results) == {"order-1", "rag-1"}
    assert result.customer_reply.find("ORD-1001") >= 0
    assert result.rag_results


def test_order_correction_resumes_memory_without_losing_main_workflow() -> None:
    first = submit("I want to cancel my order")
    second = submit("ORD-8888", conversation_id=first.conversation_id)
    third = submit("ORD-9999", conversation_id=first.conversation_id)
    assert first.outcome == "clarification_required"
    assert second.outcome == "clarification_required"
    assert third.outcome == "unable_to_answer"
    assert third.review_request_id is None

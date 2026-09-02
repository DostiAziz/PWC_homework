from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from pwc_support.domain.models import RagRequest
from pwc_support.domain.state import SupportState
from pwc_support.rag.answer import RagAnswerer
from pwc_support.workflow.policy import ReviewPolicy


def build_graph(
    *,
    policy: ReviewPolicy | None = None,
    rag_answerer: RagAnswerer | None = None,
    enable_interrupt: bool = False,
    checkpointer: Any = None,
) -> Any:
    review_policy = policy or ReviewPolicy.default()

    def intake(state: SupportState) -> dict[str, Any]:
        return {"events": [{"node": "intake", "event_type": "completed"}]}

    def triage(state: SupportState) -> dict[str, Any]:
        body = str(state.get("message", {}).get("body", ""))
        decision = review_policy.evaluate(body)
        return {
            "triage": {
                "intent": "client_enquiry",
                "review_categories": [category.value for category in decision.categories],
                "route": "review" if decision.requires_review else "plan",
                "confidence": 0.9,
            },
            "events": [{"node": "triage", "event_type": "completed"}],
        }

    def route_after_triage(state: SupportState) -> Literal["plan_work", "human_review"]:
        return "human_review" if state["triage"]["route"] == "review" else "plan_work"

    def plan_work(state: SupportState) -> dict[str, Any]:
        return {
            "plan": {
                "tasks": [{
                    "task_id": "knowledge-1",
                    "kind": "knowledge_query",
                    "input": state["message"]["body"],
                }]
            },
            "expected_task_ids": ["knowledge-1"],
            "events": [{"node": "plan_work", "event_type": "completed"}],
        }

    def case_tools(state: SupportState) -> dict[str, Any]:
        return {"events": [{"node": "case_tools", "event_type": "completed"}]}

    def compose_reply(state: SupportState) -> dict[str, Any]:
        body = state["message"]["body"]
        if rag_answerer is not None:
            result = rag_answerer.answer(RagRequest(question=body))
            return {
                "rag_results": [result.model_dump(mode="json")],
                "draft": {"text": result.answer},
                "events": [{"node": "compose_reply", "event_type": "completed"}],
            }
        return {
            "draft": {
                "text": (
                    "Based on the available public information, PwC can help "
                    f"with this enquiry: {body}"
                )
            },
            "events": [{"node": "compose_reply", "event_type": "completed"}],
        }

    def verify_response(state: SupportState) -> dict[str, Any]:
        if state.get("rag_results", [{}])[0].get("status") == "insufficient_evidence":
            return {
                "verification": {"route": "review"},
                "events": [{"node": "verify_response", "event_type": "review"}],
            }
        return {
            "verification": {"route": "release"},
            "events": [{"node": "verify_response", "event_type": "completed"}],
        }

    def route_after_verification(state: SupportState) -> Literal["finalise_case", "human_review"]:
        return "human_review" if state["verification"]["route"] == "review" else "finalise_case"

    def human_review(state: SupportState) -> dict[str, Any]:
        if enable_interrupt:
            review = interrupt({
                "type": "human_review_required",
                "categories": state.get("triage", {}).get("review_categories", []),
                "message": state["message"]["body"],
            })
            return {
                "delivery": {
                    "message": review.get("reply", "Your enquiry has been reviewed.")
                },
                "outcome": {"status": "answered"},
                "events": [{"node": "human_review", "event_type": "resumed"}],
            }
        return {
            "delivery": {
                "message": "Your enquiry has been received and is pending specialist review."
            },
            "outcome": {"status": "pending_review"},
            "events": [{"node": "human_review", "event_type": "pending"}],
        }

    def finalise_case(state: SupportState) -> dict[str, Any]:
        message = state.get("draft", {}).get("text", "")
        return {
            "delivery": {"message": message},
            "outcome": {"status": "answered"},
            "events": [{"node": "finalise_case", "event_type": "completed"}],
        }

    builder = StateGraph(SupportState)
    for name, node in (
        ("intake", intake), ("triage", triage), ("plan_work", plan_work),
        ("case_tools", case_tools), ("compose_reply", compose_reply),
        ("verify_response", verify_response), ("human_review", human_review),
        ("finalise_case", finalise_case),
    ):
        builder.add_node(name, node)
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "triage")
    builder.add_conditional_edges("triage", route_after_triage)
    builder.add_edge("plan_work", "case_tools")
    builder.add_edge("case_tools", "compose_reply")
    builder.add_edge("compose_reply", "verify_response")
    builder.add_conditional_edges("verify_response", route_after_verification)
    builder.add_edge("human_review", END)
    builder.add_edge("finalise_case", END)
    return builder.compile(checkpointer=checkpointer)

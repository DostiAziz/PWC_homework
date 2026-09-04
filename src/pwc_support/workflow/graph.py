from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Literal
from uuid import UUID, uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from pwc_support.agents.classifier import IntentClassifier
from pwc_support.agents.contracts import (
    ConversationMemory,
    SpecialistName,
    SpecialistStatus,
)
from pwc_support.domain.models import (
    Citation,
    OutcomeStatus,
    Route,
)
from pwc_support.domain.state import SupportState
from pwc_support.workflow.policy import ReviewPolicy
from pwc_support.workflow.tools import Toolbox, find_case_reference

# (Keep normalize_markers and bracket translation here)
MARKER = re.compile(r"\[S\d+\]")
LOOKALIKE_BRACKETS = str.maketrans(
    {
        "\u3010": "[",
        "\u3011": "]",
        "\uff3b": "[",
        "\uff3d": "]",
        "\u3014": "[",
        "\u3015": "]",
        "\u2768": "[",
        "\u2769": "]",
    }
)


def normalize_markers(text: str) -> str:
    return text.translate(LOOKALIKE_BRACKETS)


GREETING_REPLY = "Hello, and welcome to PwC client support. Ask me about publicly described PwC services, industries, or how the PwC network is organised, and I will answer from our published information with sources."
CLARIFICATION_REPLY = "I can help, but I need a little more detail. Please tell me which service, industry, or territory your question is about."
UNSUPPORTED_REPLY = "I could not find published information that answers this question, so I will not guess. Please rephrase it, or ask for a specialist if the matter is client specific."
PENDING_REVIEW_REPLY = "Thank you. This enquiry needs a PwC specialist, so I have logged it for review rather than answering automatically. A specialist will follow up."


def _event(node: str, event_type: str, started: float, **details: Any) -> dict[str, Any]:
    return {
        "node": node,
        "event_type": event_type,
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        "details": details,
    }


def renumber_citations(results: list[dict[str, Any]]) -> tuple[str, tuple[Citation, ...]]:
    texts: list[str] = []
    citations: list[Citation] = []
    for result in results:
        text = normalize_markers(str(result.get("answer", "")).strip())
        if not text:
            continue
        mapping: list[tuple[str, str]] = []
        for citation in result.get("citations", []):
            marker = f"[S{len(citations) + 1}]"
            citations.append(Citation.model_validate({**citation.model_dump(), "marker": marker}))
            mapping.append((str(citation.marker), marker))
        for old, new in sorted(mapping, key=lambda pair: -len(pair[0])):
            text = text.replace(old, f"\x00{new}\x00")
        texts.append(text.replace("\x00", ""))
    return "\n\n".join(texts), tuple(citations)


def build_graph(
    *,
    classifier: IntentClassifier,
    order_agent: CompiledStateGraph,
    product_agent: CompiledStateGraph,
    return_refund_agent: CompiledStateGraph,
    rag_agent: CompiledStateGraph,
    conversation_memory_repo: Any,  # We actually don't call repo here, state has conversation_memory dict
    policy: ReviewPolicy | None = None,
    toolbox: Toolbox | None = None,
    max_planned_tasks: int = 4,
) -> CompiledStateGraph:
    review_policy = policy or ReviewPolicy.default()
    tools = toolbox or Toolbox()

    def intake(state: SupportState) -> dict[str, Any]:
        started = time.perf_counter()
        message = dict(state.get("message", {}))
        body = " ".join(str(message.get("body", "")).split())
        message["body"] = body
        conversation_id = str(
            state.get("conversation_id") or message.get("conversation_id") or uuid4()
        )
        thread_id = str(state.get("thread_id") or message.get("thread_id") or conversation_id)
        client_id = str(state.get("client_id") or message.get("sender_id") or "anonymous-client")
        channel = str(state.get("channel") or message.get("channel") or "chat")
        message.setdefault("conversation_id", conversation_id)
        message.setdefault("thread_id", thread_id)
        return {
            "message": message,
            "run_id": str(state.get("run_id") or uuid4()),
            "conversation_id": conversation_id,
            "thread_id": thread_id,
            "client_id": client_id,
            "channel": channel,
            "draft_version": 1,
            "events": [
                _event("intake", "completed", started, channel=channel, characters=len(body))
            ],
        }

    def load_conversation_memory(state: SupportState) -> dict[str, Any]:
        started = time.perf_counter()
        active_specialist = None
        mem_dict = state.get("conversation_memory")
        if mem_dict:
            mem = ConversationMemory.model_validate(mem_dict)
            if mem.active_specialist:
                active_specialist = mem.active_specialist.value
        return {
            "active_specialist": active_specialist,
            "events": [
                _event(
                    "load_conversation_memory",
                    "completed",
                    started,
                    active_specialist=active_specialist,
                )
            ],
        }

    def classify_and_plan(state: SupportState) -> dict[str, Any]:
        started = time.perf_counter()
        body = str(state["message"]["body"])

        # Triage for cases
        decision = review_policy.evaluate(body)
        route = review_policy.classify_route(body)
        categories = sorted(category.value for category in decision.categories)
        snapshot = {
            "input_fingerprint": hashlib.sha256(body.encode()).hexdigest(),
            "provider_message_id": str(state.get("inbound_message_id") or state["run_id"]),
            "policy_version": decision.policy_version,
            "deterministic_match": bool(decision.matched_rule_ids),
            "matched_rule_ids": list(decision.matched_rule_ids),
            "pre_retrieval_route": route.value,
            "snapshot_hash": hashlib.sha256(body.encode()).hexdigest(),
        }

        triage_dict = {
            "intent": "client_enquiry",
            "route": route.value,
            "review_categories": categories,
            "case_reference": find_case_reference(body),
            "confidence": 0.9 if route is not Route.CLARIFY else 0.4,
        }

        # Classification
        if route in (Route.GREETING, Route.CLARIFY, Route.UNSUPPORTED):
            return {
                "triage": triage_dict,
                "routing_snapshot": snapshot,
                "events": [_event("classify_and_plan", "static_route", started, route=route.value)],
            }

        active_sp = state.get("active_specialist")
        if active_sp:
            # Route directly to active specialist for clarification
            return {
                "triage": triage_dict,
                "routing_snapshot": snapshot,
                "routing_tasks": [
                    {"task_id": "resume-1", "specialist": active_sp, "user_text": body}
                ],
                "events": [
                    _event("classify_and_plan", "resumed_specialist", started, specialist=active_sp)
                ],
            }

        routing_decision = classifier.classify(body)
        routing_tasks = [t.model_dump(mode="json") for t in routing_decision.tasks]

        return {
            "triage": triage_dict,
            "routing_snapshot": snapshot,
            "routing_decision": routing_decision.model_dump(mode="json"),
            "routing_tasks": routing_tasks,
            "events": [
                _event("classify_and_plan", "classified", started, tasks=len(routing_tasks))
            ],
        }

    def route_after_planning(
        state: SupportState,
    ) -> Literal["respond_directly"] | list[Send]:
        route = state["triage"]["route"]
        if route in (Route.GREETING.value, Route.CLARIFY.value, Route.UNSUPPORTED.value):
            return "respond_directly"
        sends = []
        for t in state.get("routing_tasks", []):
            sp = t["specialist"]
            node_name = f"{sp}_agent"
            sends.append(
                Send(
                    node_name,
                    {
                        "message": state["message"],
                        "run_id": state["run_id"],
                        "conversation_id": state["conversation_id"],
                        "client_id": state["client_id"],
                        "conversation_memory": state.get("conversation_memory"),
                        "task": t,
                    },
                )
            )
        return sends

    def join_specialist_results(state: SupportState) -> dict[str, Any]:
        started = time.perf_counter()
        return {"events": [_event("join_specialist_results", "completed", started)]}

    def compose_reply(state: SupportState) -> dict[str, Any]:
        started = time.perf_counter()
        sp_results = state.get("specialist_results", {})

        clarification_required = any(
            r.status == SpecialistStatus.NEEDS_INFORMATION for r in sp_results.values()
        )
        unable = any(r.status == SpecialistStatus.UNSUPPORTED for r in sp_results.values())

        reply_texts = []
        for task_id, res in sorted(sp_results.items()):
            if res.customer_message:
                reply_texts.append(res.customer_message)
        
        all_citations = []
        for task_id, res in sorted(sp_results.items()):
            if res.citations:
                all_citations.extend(res.citations)
        
        # Combine the texts
        draft_text = "\n\n".join(reply_texts)
        rag_results = []
        
        # Sort results by task_id to maintain deterministic order
        for task_id, res in sorted(sp_results.items()):
            if res.specialist == SpecialistName.RAG and res.customer_message:
                rag_results.append({"answer": res.customer_message, "citations": res.citations or []})

        # Renumber citations if RAG results are present
        if rag_results:
            text, citations = renumber_citations(rag_results)
            reply_texts = [
                text if res.specialist == SpecialistName.RAG else res.customer_message
                for task_id, res in sorted(sp_results.items())
                if res.customer_message
            ]
            final_citations = [c.model_dump(mode="json") for c in citations]
        else:
            final_citations = []

        final_text = "\n\n".join(reply_texts)

        outcome_status = OutcomeStatus.ANSWERED.value
        if clarification_required:
            outcome_status = OutcomeStatus.CLARIFICATION_REQUIRED.value
        elif unable and not final_text:
            outcome_status = OutcomeStatus.UNABLE_TO_ANSWER.value

        return {
            "draft": {
                "text": final_text,
                "citations": final_citations,
                "citation_markers": [c["marker"] for c in final_citations],
                "response_version": 1,
                "source": "system",
            },
            "outcome": {"status": outcome_status},
            "events": [_event("compose_reply", "completed", started)],
        }

    def prepare_escalation(state: SupportState) -> dict[str, Any]:
        started = time.perf_counter()
        categories = state["triage"].get("review_categories") or []
        sp_results = state.get("specialist_results", {})
        review_requests = [
            r for r in sp_results.values() if r.status == SpecialistStatus.REVIEW_REQUIRED
        ]

        if not categories and not review_requests:
            return {"events": [_event("prepare_escalation", "skipped", started)]}

        review_id = str(uuid4())

        # Open Case
        case_id = None
        tool_calls = []
        if tools.case_tool is not None:
            record, call = tools.case_tool.open_case(
                conversation_id=UUID(state["conversation_id"]),
                client_id=state["client_id"],
                category=str(categories[0]) if categories else "specialist_review",
                summary=str(state["message"]["body"]),
            )
            case_id = record.case_id
            tool_calls.append(call.as_event())

        proposed_actions = []
        for r in review_requests:
            if r.action_proposal:
                proposed_actions.append(r.action_proposal)

        proposed_reply = state.get("draft", {}).copy()
        proposed_reply.pop("citations", None)
        if not proposed_reply.get("text"):
            proposed_reply = None

        request = {
            "review_id": review_id,
            "case_id": case_id,
            "run_id": state["run_id"],
            "conversation_id": state["conversation_id"],
            "categories": categories,
            "original_message": state["message"]["body"],
            "proposed_reply": proposed_reply,
            "proposed_actions": proposed_actions,
            "evidence": state.get("draft", {}).get("citations", []),
            "evidence_state": "selected"
            if state.get("draft", {}).get("citations", [])
            else "unavailable",
            "routing_provenance": state.get("routing_snapshot"),
            "delivery_recipient": state.get("client_id"),
            "delivery_thread_id": state.get("thread_id"),
            "delivery_subject": state.get("message", {}).get("subject") or "Re: your enquiry",
            "response_version": 1,
        }
        return {
            "review_request": request,
            "case_id": case_id,
            "tool_calls": tool_calls,
            "outcome": {"status": OutcomeStatus.PENDING_REVIEW.value, "review_id": review_id},
            "draft": {
                "text": PENDING_REVIEW_REPLY,
                "citations": [],
                "citation_markers": [],
                "response_version": 1,
                "source": "system",
            },
            "events": [_event("prepare_escalation", "completed", started, review_id=review_id)],
        }

    def respond_directly(state: SupportState) -> dict[str, Any]:
        started = time.perf_counter()
        route = state["triage"]["route"]
        if route == Route.GREETING.value:
            message, status = GREETING_REPLY, OutcomeStatus.ANSWERED
        elif route == Route.CLARIFY.value:
            message, status = CLARIFICATION_REPLY, OutcomeStatus.CLARIFICATION_REQUIRED
        else:
            message, status = UNSUPPORTED_REPLY, OutcomeStatus.UNABLE_TO_ANSWER
        return {
            "draft": {
                "text": message,
                "citation_markers": [],
                "citations": [],
                "response_version": 1,
                "source": "system",
            },
            "outcome": {"status": status.value},
            "events": [_event("respond_directly", status.value, started, route=route)],
        }

    def finalise(state: SupportState) -> dict[str, Any]:
        started = time.perf_counter()
        message = str(state.get("draft", {}).get("text", "")) or UNSUPPORTED_REPLY
        status = OutcomeStatus(state.get("outcome", {}).get("status", OutcomeStatus.ANSWERED.value))
        calls: list[dict[str, Any]] = []
        case_id = state.get("case_id")

        channel = state.get("channel", "chat")
        
        if channel == "simulated_email" and toolbox and toolbox.mailbox_tool:
            subject = state.get("message", {}).get("subject") or "Your enquiry"
            result = toolbox.mailbox_tool.deliver(
                recipient=state.get("client_id", ""),
                subject=subject,
                body=message,
                thread_id=state.get("thread_id") or "",
            )
            calls.append({"tool": "simulated_mailbox", "result": result})
        return {
            "delivery": {
                "message": message,
                "channel": state.get("channel", "chat"),
                "thread_id": state.get("thread_id"),
                "citations": state.get("draft", {}).get("citations", []),
            },
            "outcome": {**state.get("outcome", {}), "status": status.value, "case_id": case_id},
            "tool_calls": calls,
            "events": [_event("finalise", status.value, started, case_id=case_id)],
        }

    builder = StateGraph(SupportState)
    builder.add_node("intake", intake)
    builder.add_node("load_conversation_memory", load_conversation_memory)
    builder.add_node("classify_and_plan", classify_and_plan)

    # Add agent nodes
    def call_agent(agent_name, agent_graph):
        def _call(state):
            # In LangGraph, dispatching via Send maps to a state update.
            # We must run the specialist graph and return its output.
            res = agent_graph.invoke(state)
            return {"specialist_results": res.get("specialist_results", {})}

        return _call

    builder.add_node("order_agent", call_agent("order_agent", order_agent))
    builder.add_node("product_agent", call_agent("product_agent", product_agent))
    builder.add_node("return_refund_agent", call_agent("return_refund_agent", return_refund_agent))
    builder.add_node("rag_agent", call_agent("rag_agent", rag_agent))

    builder.add_node("join_specialist_results", join_specialist_results)
    builder.add_node("compose_reply", compose_reply)
    builder.add_node("prepare_escalation", prepare_escalation)
    builder.add_node("respond_directly", respond_directly)
    builder.add_node("finalise", finalise)

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "load_conversation_memory")
    builder.add_edge("load_conversation_memory", "classify_and_plan")

    builder.add_conditional_edges(
        "classify_and_plan", route_after_planning, ["order_agent", "product_agent", "return_refund_agent", "rag_agent", "respond_directly"]
    )

    builder.add_edge("order_agent", "join_specialist_results")
    builder.add_edge("product_agent", "join_specialist_results")
    builder.add_edge("return_refund_agent", "join_specialist_results")
    builder.add_edge("rag_agent", "join_specialist_results")

    builder.add_edge("join_specialist_results", "compose_reply")
    builder.add_edge("compose_reply", "prepare_escalation")
    builder.add_edge("prepare_escalation", "finalise")
    builder.add_edge("respond_directly", "finalise")
    builder.add_edge("finalise", END)

    return builder.compile()

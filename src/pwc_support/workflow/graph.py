from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Literal
from uuid import UUID, uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from pwc_support.domain.models import (
    CaseStatus,
    Citation,
    DraftReply,
    OutcomeStatus,
    PlannedTask,
    ProposedAction,
    RagRequest,
    ReviewDecisionKind,
    Route,
    TaskKind,
    TaskResult,
    TaskStatus,
    Verification,
    WorkPlan,
)
from pwc_support.domain.state import SupportState
from pwc_support.rag.answer import RagAnswerer
from pwc_support.rag.subgraph import to_rag_result
from pwc_support.workflow.policy import ReviewPolicy
from pwc_support.workflow.policy import merge_routing
from pwc_support.domain.errors import RiskClassificationUnavailable
from pwc_support.workflow.tools import Toolbox, find_case_reference

MARKER = re.compile(r"\[S\d+\]")
# gpt-oss and other local models often emit lookalike bracket pairs around a citation
# marker. Normalising them keeps a genuinely grounded answer from being withheld, while
# still letting verification catch markers that match no retrieved source.
LOOKALIKE_BRACKETS = str.maketrans(
    {
        "\u3010": "[", "\u3011": "]",  # CJK lenticular brackets
        "\uff3b": "[", "\uff3d": "]",  # fullwidth square brackets
        "\u3014": "[", "\u3015": "]",  # tortoise shell brackets
        "\u2768": "[", "\u2769": "]",  # medium parenthesis ornaments
    }
)


def normalize_markers(text: str) -> str:
    """Rewrite lookalike bracket pairs so citation markers are recognisable ASCII."""
    return text.translate(LOOKALIKE_BRACKETS)

GREETING_REPLY = (
    "Hello, and welcome to PwC client support. Ask me about publicly described PwC "
    "services, industries, or how the PwC network is organised, and I will answer from "
    "our published information with sources."
)
CLARIFICATION_REPLY = (
    "I can help, but I need a little more detail. Please tell me which service, "
    "industry, or territory your question is about."
)
UNSUPPORTED_REPLY = (
    "I could not find published information that answers this question, so I will not "
    "guess. Please rephrase it, or ask for a specialist if the matter is client specific."
)
PENDING_REVIEW_REPLY = (
    "Thank you. This enquiry needs a PwC specialist, so I have logged it for review "
    "rather than answering automatically. A specialist will follow up."
)
REJECTED_REPLY = (
    "A PwC specialist has reviewed this enquiry and cannot respond through this channel. "
    "Please contact your engagement team directly."
)


def _event(node: str, event_type: str, started: float, **details: Any) -> dict[str, Any]:
    return {
        "node": node,
        "event_type": event_type,
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        "details": details,
    }


def split_questions(body: str, *, limit: int) -> list[str]:
    """Split a multi-part enquiry so each question can be retrieved independently."""
    parts = [part.strip() for part in re.split(r"(?<=\?)\s+", body) if part.strip()]
    substantive = [part for part in parts if len(re.findall(r"[A-Za-z0-9]+", part)) >= 3]
    if len(substantive) < 2:
        return [body]
    return substantive[:limit]


def renumber_citations(
    results: list[dict[str, Any]],
) -> tuple[str, tuple[Citation, ...]]:
    """Merge per-task answers into one reply with a single, non-colliding marker series."""
    texts: list[str] = []
    citations: list[Citation] = []
    for result in results:
        text = normalize_markers(str(result.get("answer", "")).strip())
        if not text:
            continue
        mapping: list[tuple[str, str]] = []
        for citation in result.get("citations", []):
            marker = f"[S{len(citations) + 1}]"
            citations.append(Citation.model_validate({**citation, "marker": marker}))
            mapping.append((str(citation["marker"]), marker))
        for old, new in sorted(mapping, key=lambda pair: -len(pair[0])):
            text = text.replace(old, f"\x00{new}\x00")
        texts.append(text.replace("\x00", ""))
    return "\n\n".join(texts), tuple(citations)


def build_graph(
    *,
    policy: ReviewPolicy | None = None,
    risk_classifier: Any = None,
    rag_answerer: RagAnswerer | None = None,
    toolbox: Toolbox | None = None,
    retail_tools: list[Any] | None = None,
    enable_interrupt: bool = False,
    checkpointer: Any = None,
    max_planned_tasks: int = 4,
) -> Any:
    """Compile the main support workflow: triage, decomposition, tools, review, delivery."""
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
            "events": [_event("intake", "completed", started, channel=channel,
                              characters=len(body))],
        }

    def triage(state: SupportState) -> dict[str, Any]:
        started = time.perf_counter()
        body = str(state["message"]["body"])
        decision = review_policy.evaluate(body)
        route = review_policy.classify_route(body)
        retail_lookup = any(word in body.casefold() for word in ("product", "jacket", "backpack", "headphones", "offer"))
        retail_action = any(word in body.casefold() for word in ("return", "refund", "send back"))
        if retail_lookup and not retail_action and not decision.requires_review:
            route = Route.PLAN
        semantic = None
        failure_class = None
        if not decision.requires_review and route is Route.PLAN and risk_classifier is not None:
            try:
                semantic = risk_classifier.classify(enquiry=body)
                requires_review, merged_categories = merge_routing(
                    deterministic=decision, semantic=semantic
                )
                if requires_review:
                    route = Route.REVIEW
                    decision = decision.__class__(True, merged_categories, (), decision.policy_version)
            except RiskClassificationUnavailable as error:
                failure_class = error.failure_class
                route = Route.UNSUPPORTED
        categories = sorted(category.value for category in decision.categories)
        snapshot_payload = {
            "body": body,
            "route": route.value,
            "categories": categories,
            "semantic": semantic.model_dump(mode="json") if semantic else None,
            "failure_class": failure_class,
        }
        snapshot_hash = hashlib.sha256(json.dumps(snapshot_payload, sort_keys=True).encode()).hexdigest()
        snapshot = {
            "input_fingerprint": hashlib.sha256(body.encode()).hexdigest(),
            "provider_message_id": str(state.get("inbound_message_id") or state["run_id"]),
            "policy_version": decision.policy_version,
            "deterministic_match": bool(decision.matched_rule_ids),
            "matched_rule_ids": list(decision.matched_rule_ids),
            "classifier_invoked": semantic is not None,
            "classifier_attempts": 1 if semantic is not None else 0,
            "classifier": semantic.model_dump(mode="json") if semantic else None,
            "failure_class": failure_class,
            "pre_retrieval_route": route.value,
            "snapshot_hash": snapshot_hash,
        }
        return {
            "triage": {
                "intent": "client_enquiry",
                "route": route.value,
                "review_categories": categories,
                "case_reference": find_case_reference(body),
                "confidence": 0.9 if route is not Route.CLARIFY else 0.4,
            },
            "routing_snapshot": snapshot,
            "events": [_event("triage", "completed", started, route=route.value,
                              review_categories=categories)],
        }

    def route_after_triage(
        state: SupportState,
    ) -> Literal["plan_work", "gather_evidence", "respond_directly"]:
        route = state["triage"]["route"]
        if route == Route.REVIEW.value:
            return "gather_evidence"
        if route in (Route.GREETING.value, Route.CLARIFY.value, Route.UNSUPPORTED.value):
            return "respond_directly"
        return "plan_work"

    def gather_evidence(state: SupportState) -> dict[str, Any]:
        """Research an escalated enquiry for the specialist without drafting a reply.

        A sensitive enquiry must never have the model write an answer, so this runs the
        RAG subgraph in evidence-only mode: the specialist opens the review with the
        relevant published sources already in front of them instead of a blank box.
        """
        started = time.perf_counter()
        body = str(state["message"]["body"])
        if retail_tools and any(word in body.casefold() for word in ("return", "refund")) and len(retail_tools) > 3:
            order_match = re.search(r"\bORD-[A-Z0-9-]+\b", body, re.IGNORECASE)
            item_match = re.search(r"\bITEM-[A-Z0-9-]+\b", body, re.IGNORECASE)
            if order_match and item_match:
                action = retail_tools[3].invoke({
                    "order_id": order_match.group(0).upper(),
                    "item_id": item_match.group(0).upper(),
                    "reason": body,
                    "idempotency_key": f"chat-{state['run_id']}",
                })
                return {
                    "action_proposal": action,
                    "events": [_event("gather_evidence", "retail_return_evaluated", started, status=(action.get("return") or {}).get("status"))],
                }
        if rag_answerer is None:
            return {
                "events": [_event("gather_evidence", "skipped", started,
                                  reason="no_retrieval_runtime")]
            }
        bundle = rag_answerer.gather_evidence(RagRequest(question=body))
        return {
            "evidence": bundle.model_dump(mode="json"),
            "events": [_event("gather_evidence", "completed", started,
                              sources=len(bundle.citations))],
        }

    def plan_work(state: SupportState) -> dict[str, Any]:
        started = time.perf_counter()
        body = str(state["message"]["body"])
        if retail_tools and any(word in body.casefold() for word in ("product", "jacket", "backpack", "headphones", "offer")):
            query = next((word for word in ("jacket", "backpack", "headphones", "offer", "product") if word in body.casefold()), body)
            result = retail_tools[0].invoke({"query": query})
            products = result.get("products", []) if isinstance(result, dict) else []
            answer = "\n".join(f"{item['name']}: {item['price']} {item['currency']} ({item['stock']} in stock)" for item in products) or "No matching products found."
            return {"plan": {"tasks": []}, "expected_task_ids": [], "retail_intent": "product_search", "retail_answer": answer, "events": [_event("plan_work", "retail_lookup", started, count=len(products))]}
        reference = state["triage"].get("case_reference")
        knowledge_limit = max_planned_tasks - (1 if reference else 0)
        tasks = [
            PlannedTask(
                task_id=f"knowledge-{index}",
                kind=TaskKind.KNOWLEDGE_QUERY,
                input=question[:2000],
            )
            for index, question in enumerate(
                split_questions(body, limit=knowledge_limit), start=1
            )
        ]
        if reference:
            tasks.append(
                PlannedTask(task_id="case-1", kind=TaskKind.CASE_LOOKUP, input=reference)
            )
        plan = WorkPlan(tasks=tuple(tasks))
        return {
            "plan": plan.model_dump(mode="json"),
            "expected_task_ids": [task.task_id for task in plan.tasks],
            "events": [_event("plan_work", "completed", started,
                              task_ids=[task.task_id for task in plan.tasks],
                              task_kinds=[task.kind.value for task in plan.tasks])],
        }

    def fan_out_tasks(state: SupportState) -> list[Send] | str:
        """Dispatch every planned task to an independent worker in one superstep."""
        if state.get("retail_intent") == "product_search":
            return "compose_reply"
        return [
            Send(
                "execute_task",
                {
                    "task": task,
                    "message": state["message"],
                    "run_id": state["run_id"],
                    "conversation_id": state["conversation_id"],
                    "client_id": state["client_id"],
                },
            )
            for task in state["plan"]["tasks"]
        ]

    def execute_task(state: SupportState) -> dict[str, Any]:
        started = time.perf_counter()
        task = PlannedTask.model_validate(state["task"])
        if task.kind is TaskKind.CASE_LOOKUP:
            return _execute_case_lookup(task, started)
        return _execute_knowledge_query(task, started)

    def _execute_case_lookup(task: PlannedTask, started: float) -> dict[str, Any]:
        if tools.case_tool is None:
            result = TaskResult(
                task_id=task.task_id, status=TaskStatus.SKIPPED, error_code="TOOL_UNAVAILABLE"
            )
            return {
                "task_results": {task.task_id: result},
                "events": [_event("execute_task", "skipped", started, task_id=task.task_id)],
            }
        lookup, call = tools.case_tool.lookup(task.input)
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.SUCCESS if lookup.found else TaskStatus.FAILURE,
            payload=lookup.model_dump(mode="json"),
            error_code=None if lookup.found else "CASE_NOT_FOUND",
        )
        return {
            "task_results": {task.task_id: result},
            "tool_calls": [call.as_event()],
            "events": [_event("execute_task", "completed", started, task_id=task.task_id,
                              kind=task.kind.value, found=lookup.found)],
        }

    def _execute_knowledge_query(task: PlannedTask, started: float) -> dict[str, Any]:
        if rag_answerer is None:
            result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.SKIPPED,
                error_code="RETRIEVAL_UNAVAILABLE",
            )
            return {
                "task_results": {task.task_id: result},
                "events": [_event("execute_task", "skipped", started, task_id=task.task_id,
                                  reason="no_retrieval_runtime")],
            }
        rag_state = rag_answerer.run(RagRequest(question=task.input))
        rag_result = to_rag_result(rag_state)
        timings = {
            f"rag.{node}": value
            for node, value in rag_state.get("node_timings", {}).items()
        }
        payload = rag_result.model_dump(mode="json")
        payload["task_id"] = task.task_id
        payload["node_timings"] = timings
        answered = rag_result.status == "answered"
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.SUCCESS if answered else TaskStatus.FAILURE,
            payload={"status": rag_result.status, "citations": len(rag_result.citations)},
            error_code=None if answered else "EVIDENCE_INSUFFICIENT",
        )
        return {
            "task_results": {task.task_id: result},
            "rag_results": [payload],
            "events": [_event("execute_task", "completed", started, task_id=task.task_id,
                              kind=task.kind.value, status=rag_result.status,
                              citations=len(rag_result.citations), **timings)],
        }

    def case_tools(state: SupportState) -> dict[str, Any]:
        """Join the fan-out and record the enquiry in the durable case system."""
        started = time.perf_counter()
        if tools.case_tool is None:
            return {"events": [_event("case_tools", "skipped", started,
                                      reason="no_case_tool")]}
        existing = _resolved_case_id(state)
        if existing is not None:
            return {
                "case_id": existing,
                "events": [_event("case_tools", "completed", started, case_id=existing,
                                  operation="reused")],
            }
        categories = state["triage"].get("review_categories") or []
        if not categories and not state["triage"].get("case_reference"):
            return {
                "events": [_event("case_tools", "skipped", started,
                                  reason="routine_enquiry_no_case")]
            }
        record, call = tools.case_tool.open_case(
            conversation_id=UUID(state["conversation_id"]),
            client_id=state["client_id"],
            category=str(categories[0]) if categories else "general_enquiry",
            summary=str(state["message"]["body"]),
        )
        return {
            "case_id": record.case_id,
            "tool_calls": [call.as_event()],
            "proposed_actions": [
                ProposedAction(
                    action_type="create_case",
                    description=f"Track this enquiry as case {record.case_id}.",
                    requires_review=False,
                ).model_dump(mode="json")
            ],
            "events": [_event("case_tools", "completed", started, case_id=record.case_id,
                              operation="opened")],
        }

    def _resolved_case_id(state: SupportState) -> str | None:
        for result in state.get("task_results", {}).values():
            if result.status is TaskStatus.SUCCESS and result.payload.get("found"):
                case = result.payload.get("case") or {}
                case_id = case.get("case_id")
                if case_id:
                    return str(case_id)
        return None

    def compose_reply(state: SupportState) -> dict[str, Any]:
        started = time.perf_counter()
        if state.get("retail_intent") == "product_search":
            return {
                "draft": {"text": str(state.get("retail_answer", "No matching products found.")), "citation_markers": [], "citations": [], "response_version": 1, "source": "system"},
                "events": [_event("compose_reply", "retail", started)],
            }
        answered = [
            result
            for result in state.get("rag_results", [])
            if result.get("status") == "answered"
        ]
        text, citations = renumber_citations(answered)
        if not text:
            return {
                "draft": {"text": "", "citation_markers": [], "response_version": 1},
                "events": [_event("compose_reply", "insufficient_evidence", started,
                                  knowledge_results=len(state.get("rag_results", [])))],
            }
        draft = DraftReply(
            text=text[:1500],
            citation_markers=tuple(citation.marker for citation in citations),
            response_version=int(state.get("draft_version", 1)),
        )
        return {
            "draft": {
                **draft.model_dump(mode="json"),
                "citations": [citation.model_dump(mode="json") for citation in citations],
            },
            "events": [_event("compose_reply", "completed", started,
                              citations=len(citations), characters=len(draft.text))],
        }

    def verify_response(state: SupportState) -> dict[str, Any]:
        """Reject ungrounded drafts: no evidence, no citations, or invented markers."""
        started = time.perf_counter()
        draft = state.get("draft", {})
        text = str(draft.get("text", ""))
        citations = draft.get("citations", [])
        if state.get("retail_intent") == "product_search":
            verification = Verification(route="release", reason="Structured product facts supplied by database tool")
            return {"verification": verification.model_dump(mode="json"), "events": [_event("verify_response", "release", started, citations=0)]}
        known = {str(citation["marker"]) for citation in citations}
        invented = sorted(set(MARKER.findall(text)) - known)
        if not text or not citations:
            verification = Verification(
                route="review" if _needs_specialist(state) else "revise",
                error_code="EVIDENCE_INSUFFICIENT",
                reason="No grounded evidence was selected for this enquiry.",
            )
        elif invented:
            verification = Verification(
                route="review",
                error_code="MODEL_OUTPUT_INVALID",
                reason=f"Draft cites unknown sources: {', '.join(invented)}.",
            )
        elif not MARKER.search(text):
            verification = Verification(
                route="revise",
                error_code="MODEL_OUTPUT_INVALID",
                reason="Draft carries no citation marker, so its claims are unattributed.",
            )
        else:
            verification = Verification(
                route="release",
                citations=tuple(Citation.model_validate(item) for item in citations),
            )
        return {
            "verification": verification.model_dump(mode="json"),
            "events": [_event("verify_response", verification.route, started,
                              error_code=verification.error_code,
                              citations=len(citations))],
        }

    def _needs_specialist(state: SupportState) -> bool:
        """Escalate an evidence failure only when the enquiry is genuinely consequential."""
        if state["triage"].get("review_categories"):
            return True
        return _resolved_case_id(state) is not None

    def route_after_verification(
        state: SupportState,
    ) -> Literal["finalise_case", "human_review", "respond_directly"]:
        route = state["verification"]["route"]
        if route == "release":
            return "finalise_case"
        return "human_review" if route == "review" else "respond_directly"

    def respond_directly(state: SupportState) -> dict[str, Any]:
        """Deterministic replies that must never consume retrieval or model capacity."""
        started = time.perf_counter()
        route = state["triage"]["route"]
        if state.get("verification", {}).get("route") == "revise":
            message, status = UNSUPPORTED_REPLY, OutcomeStatus.UNABLE_TO_ANSWER
        elif route == Route.GREETING.value:
            message, status = GREETING_REPLY, OutcomeStatus.ANSWERED
        elif route == Route.CLARIFY.value:
            message, status = CLARIFICATION_REPLY, OutcomeStatus.CLARIFICATION_REQUIRED
        else:
            message, status = UNSUPPORTED_REPLY, OutcomeStatus.UNABLE_TO_ANSWER
        return {
            "draft": {"text": message, "citation_markers": [], "citations": [],
                      "response_version": 1, "source": "system"},
            "outcome": {"status": status.value},
            "events": [_event("respond_directly", status.value, started, route=route)],
        }

    def human_review(state: SupportState) -> dict[str, Any]:
        started = time.perf_counter()
        review_id = str(state.get("review_request", {}).get("review_id") or uuid4())
        # An escalation reaching a specialist must be a tracked case, even when triage
        # routed here directly and the case tool node was never visited.
        case_id, opened = _ensure_case(state)
        request = {
            "review_id": review_id,
            "case_id": case_id,
            "run_id": state["run_id"],
            "conversation_id": state["conversation_id"],
            "categories": state["triage"].get("review_categories", []),
            "original_message": state["message"]["body"],
            "proposed_reply": state.get("draft", {}),
            "proposed_actions": state.get("proposed_actions", []) + ([{"action_type": "retail_return", "description": "Return eligibility and refund proposal", "requires_review": True, "details": state["action_proposal"]}] if state.get("action_proposal") else []),
            # Only the triage path carries background evidence; an enquiry escalated
            # after drafting arrives with a draft that already cites its own sources.
            "evidence": state.get("evidence", {}).get("citations", []),
            "verification": state.get("verification", {}),
            "response_version": int(state.get("draft_version", 1)),
        }
        if not enable_interrupt:
            return {
                "review_request": request,
                "case_id": case_id,
                "tool_calls": opened,
                "draft": {"text": PENDING_REVIEW_REPLY, "citations": [],
                          "citation_markers": [], "response_version": 1, "source": "system"},
                "outcome": {"status": OutcomeStatus.PENDING_REVIEW.value,
                            "review_id": review_id},
                "events": [_event("human_review", "pending", started, review_id=review_id)],
            }
        decision = interrupt({"type": "human_review_required", **request})
        return {
            "review_request": request,
            "case_id": case_id,
            "tool_calls": opened,
            **_apply_review_decision(state, review_id, decision, started),
        }

    def _ensure_case(state: SupportState) -> tuple[str | None, list[dict[str, Any]]]:
        existing = state.get("case_id")
        if existing or tools.case_tool is None:
            return (str(existing) if existing else None), []
        categories = state["triage"].get("review_categories") or []
        record, call = tools.case_tool.open_case(
            conversation_id=UUID(state["conversation_id"]),
            client_id=state["client_id"],
            category=str(categories[0]) if categories else "specialist_review",
            summary=str(state["message"]["body"]),
        )
        return record.case_id, [call.as_event()]

    def _apply_review_decision(
        state: SupportState, review_id: str, decision: Any, started: float
    ) -> dict[str, Any]:
        payload = decision if isinstance(decision, dict) else {"kind": str(decision)}
        raw_kind = str(payload.get("kind") or payload.get("decision") or "approve")
        try:
            kind = ReviewDecisionKind(raw_kind)
        except ValueError:
            kind = ReviewDecisionKind.APPROVE
        proposed = str(state.get("draft", {}).get("text", "")).strip()
        edited = str(payload.get("edited_text") or payload.get("reply") or "").strip()
        if kind is ReviewDecisionKind.REJECT:
            message, status = REJECTED_REPLY, OutcomeStatus.UNABLE_TO_ANSWER
        elif kind in (ReviewDecisionKind.TAKE_OWNERSHIP, ReviewDecisionKind.REQUEST_REVISION):
            # No redraft loop exists, so asking for a revision leaves the enquiry with the
            # specialist. It must never fall through to approving the draft it rejected.
            message = edited or PENDING_REVIEW_REPLY
            status = OutcomeStatus.PENDING_REVIEW
        elif kind is ReviewDecisionKind.EDIT:
            message, status = edited or proposed, OutcomeStatus.ANSWERED
        else:
            message, status = edited or proposed, OutcomeStatus.ANSWERED
        if not message:
            message, status = PENDING_REVIEW_REPLY, OutcomeStatus.PENDING_REVIEW
        return {
            "review_decision": {
                "review_id": review_id,
                "kind": kind.value,
                "reviewer_id": str(payload.get("reviewer_id", "specialist")),
                "note": str(payload.get("note", "")),
            },
            "draft": {"text": message, "citations": state.get("draft", {}).get("citations", []),
                      "citation_markers": [], "response_version": 1, "source": "human"},
            "outcome": {"status": status.value, "review_id": review_id},
            "events": [_event("human_review", "resumed", started, review_id=review_id,
                              kind=kind.value)],
        }

    def finalise_case(state: SupportState) -> dict[str, Any]:
        """Close the case and deliver the reply through the channel's own tool."""
        started = time.perf_counter()
        message = str(state.get("draft", {}).get("text", "")) or UNSUPPORTED_REPLY
        status = OutcomeStatus(
            state.get("outcome", {}).get("status", OutcomeStatus.ANSWERED.value)
        )
        calls: list[dict[str, Any]] = []
        case_id = state.get("case_id")
        if tools.case_tool is not None and case_id:
            calls.append(tools.case_tool.close_case(case_id, _case_status(status)).as_event())
        if tools.mailbox_tool is not None and state.get("channel") == "simulated_email":
            calls.append(
                tools.mailbox_tool.deliver(
                    thread_id=str(state["thread_id"]),
                    recipient=str(state["client_id"]),
                    subject=str(state["message"].get("subject") or "Re: your enquiry"),
                    body=message,
                ).as_event()
            )
        return {
            "delivery": {
                "message": message,
                "channel": state.get("channel", "chat"),
                "thread_id": state.get("thread_id"),
                "citations": state.get("draft", {}).get("citations", []),
            },
            "outcome": {**state.get("outcome", {}), "status": status.value,
                        "case_id": case_id},
            "tool_calls": calls,
            "events": [_event("finalise_case", status.value, started, case_id=case_id,
                              delivered=bool(calls))],
        }

    def _case_status(status: OutcomeStatus) -> CaseStatus:
        if status is OutcomeStatus.PENDING_REVIEW:
            return CaseStatus.PENDING_REVIEW
        if status is OutcomeStatus.UNABLE_TO_ANSWER:
            return CaseStatus.HUMAN_OWNED
        return CaseStatus.RESOLVED

    builder = StateGraph(SupportState)
    for name, node in (
        ("intake", intake),
        ("triage", triage),
        ("gather_evidence", gather_evidence),
        ("plan_work", plan_work),
        ("execute_task", execute_task),
        ("case_tools", case_tools),
        ("compose_reply", compose_reply),
        ("verify_response", verify_response),
        ("respond_directly", respond_directly),
        ("human_review", human_review),
        ("finalise_case", finalise_case),
    ):
        builder.add_node(name, node)
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "triage")
    builder.add_conditional_edges(
        "triage", route_after_triage, ["plan_work", "gather_evidence", "respond_directly"]
    )
    builder.add_edge("gather_evidence", "human_review")
    builder.add_conditional_edges("plan_work", fan_out_tasks, ["execute_task", "compose_reply"])
    builder.add_edge("execute_task", "case_tools")
    builder.add_edge("case_tools", "compose_reply")
    builder.add_edge("compose_reply", "verify_response")
    builder.add_conditional_edges(
        "verify_response",
        route_after_verification,
        ["finalise_case", "human_review", "respond_directly"],
    )
    builder.add_edge("respond_directly", "finalise_case")
    builder.add_edge("human_review", "finalise_case")
    builder.add_edge("finalise_case", END)
    return builder.compile(checkpointer=checkpointer)

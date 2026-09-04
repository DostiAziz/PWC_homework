from __future__ import annotations

import sqlite3
import time
from typing import Any, Protocol

from chromadb.errors import ChromaError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from pwc_support.domain.models import (
    OrderAction,
    RagRequest,
    RagResult,
    Task,
    TaskKind,
    TaskResult,
    TraceEvent,
)
from pwc_support.domain.state import SupportState
from pwc_support.llm.ollama import OllamaUnavailable
from pwc_support.storage.retail_repositories import CancellationConflict
from pwc_support.workflow.commerce import Commerce
from pwc_support.workflow.planner import (
    Planner,
    PlanningUnavailable,
    is_greeting,
    parse_confirmation,
)

GREETING_REPLY = (
    "Hello. Ask me about products, offers, shipping, warranty, an order, "
    "or cancelling an eligible order."
)


class Rag(Protocol):
    def answer(self, request: RagRequest) -> RagResult: ...


def _event(node: str, event_type: str, started: float) -> TraceEvent:
    return TraceEvent(
        node=node,
        event_type=event_type,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )


def build_graph(
    *,
    planner: Planner,
    commerce: Commerce,
    rag_answerer: Rag,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    def intake(state: SupportState) -> dict[str, Any]:
        started = time.perf_counter()
        message = " ".join(state.get("message", "").split())
        update: dict[str, Any] = {
            "message": message,
            "events": [_event("intake", "normalized", started)],
        }
        if not message:
            update["direct_response"] = "Please enter a question."
        return update

    def plan_tasks(state: SupportState) -> dict[str, Any]:
        started = time.perf_counter()
        if state.get("direct_response"):
            return {"tasks": (), "events": [_event("plan_tasks", "skipped", started)]}
        message = state["message"]
        pending = state.get("pending_cancellation")
        if pending is not None:
            task = Task(
                task_id="task-1",
                kind=TaskKind.ORDER,
                request=message,
                order_id=pending.order_id,
                order_action=OrderAction.CANCEL,
            )
            return {
                "tasks": (task,),
                "confirmation": parse_confirmation(message),
                "events": [_event("plan_tasks", "confirmation", started)],
            }
        if is_greeting(message):
            return {
                "tasks": (),
                "direct_response": GREETING_REPLY,
                "events": [_event("plan_tasks", "greeting", started)],
            }
        try:
            tasks = planner.plan(message)
        except PlanningUnavailable:
            return {
                "tasks": (),
                "direct_response": (
                    "The local model could not classify that request. Please try again."
                ),
                "events": [_event("plan_tasks", "unavailable", started)],
            }
        return {"tasks": tasks, "events": [_event("plan_tasks", "planned", started)]}

    def route_tasks(state: SupportState) -> str | list[Send]:
        tasks = state.get("tasks", ())
        if not tasks:
            return "respond"
        destinations = {
            TaskKind.KNOWLEDGE: "rag_task",
            TaskKind.CATALOGUE: "catalogue_task",
            TaskKind.ORDER: "order_task",
        }
        return [
            Send(
                destinations[task.kind],
                {
                    "message": state["message"],
                    "customer_id": state["customer_id"],
                    "task": task,
                    "pending_cancellation": state.get("pending_cancellation"),
                    "confirmation": state.get("confirmation"),
                },
            )
            for task in tasks
        ]

    def rag_task(state: SupportState) -> dict[str, Any]:
        started = time.perf_counter()
        task = state["task"]
        try:
            rag = rag_answerer.answer(RagRequest(question=task.request))
            result = TaskResult(
                task_id=task.task_id,
                kind=task.kind,
                message=(
                    rag.answer
                    if rag.status == "answered"
                    else "I could not find grounded policy information for that question."
                ),
                citations=rag.citations,
            )
            event_type: str = rag.status
        except OllamaUnavailable:
            result = TaskResult(
                task_id=task.task_id,
                kind=task.kind,
                message="The local model is unavailable, so I cannot answer that policy question.",
            )
            event_type = "unavailable"
        except (ChromaError, sqlite3.Error):
            result = TaskResult(
                task_id=task.task_id,
                kind=task.kind,
                message=(
                    "The knowledge base is unavailable, so I cannot answer that policy question."
                ),
            )
            event_type = "unavailable"
        return {
            "results": [result],
            "events": [_event("rag_task", event_type, started)],
        }

    def catalogue_task(state: SupportState) -> dict[str, Any]:
        started = time.perf_counter()
        task = state["task"]
        try:
            result = commerce.catalogue(task)
            event_type = str(task.catalogue_action)
        except sqlite3.Error:
            result = TaskResult(
                task_id=task.task_id,
                kind=task.kind,
                message="The catalogue database is unavailable. Please try again.",
            )
            event_type = "unavailable"
        return {
            "results": [result],
            "events": [_event("catalogue_task", event_type, started)],
        }

    def order_task(state: SupportState) -> dict[str, Any]:
        started = time.perf_counter()
        task = state["task"]
        try:
            result = commerce.order(
                task,
                customer_id=state["customer_id"],
                pending_cancellation=state.get("pending_cancellation"),
                confirmed=state.get("confirmation"),
            )
            event_type = str(task.order_action)
        except (sqlite3.Error, CancellationConflict):
            result = TaskResult(
                task_id=task.task_id,
                kind=task.kind,
                message=(
                    "The order could not be changed safely. No cancellation was claimed. "
                    "Please check the order and try again."
                ),
            )
            event_type = "failed"
        return {
            "results": [result],
            "events": [_event("order_task", event_type, started)],
        }

    def join_results(state: SupportState) -> dict[str, Any]:
        started = time.perf_counter()
        ordered = tuple(sorted(state.get("results", []), key=lambda item: item.task_id))
        return {
            "ordered_results": ordered,
            "events": [_event("join_results", "joined", started)],
        }

    def respond(state: SupportState) -> dict[str, Any]:
        started = time.perf_counter()
        results = state.get("ordered_results", ())
        response = state.get("direct_response") or "\n\n".join(result.message for result in results)
        citations = tuple(citation for result in results for citation in result.citations)
        pending = state.get("pending_cancellation")
        if any(result.clear_pending for result in results):
            pending = None
        else:
            pending = next(
                (
                    result.pending_cancellation
                    for result in results
                    if result.pending_cancellation is not None
                ),
                pending,
            )
        return {
            "response": response,
            "citations": citations,
            "pending_cancellation": pending,
            "events": [_event("respond", "completed", started)],
        }

    builder: StateGraph[SupportState, None, SupportState, SupportState] = StateGraph(SupportState)
    builder.add_node("intake", intake)
    builder.add_node("plan_tasks", plan_tasks)
    builder.add_node("rag_task", rag_task)
    builder.add_node("catalogue_task", catalogue_task)
    builder.add_node("order_task", order_task)
    builder.add_node("join_results", join_results)
    builder.add_node("respond", respond)
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "plan_tasks")
    builder.add_conditional_edges("plan_tasks", route_tasks)
    builder.add_edge("rag_task", "join_results")
    builder.add_edge("catalogue_task", "join_results")
    builder.add_edge("order_task", "join_results")
    builder.add_edge("join_results", "respond")
    builder.add_edge("respond", END)
    return builder.compile()

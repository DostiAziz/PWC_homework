from __future__ import annotations

import re
from operator import add
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Overwrite, interrupt

from domain.models import Citation, PendingCancellation
from workflow.tools import TOOL_SCHEMAS, ToolRegistry

SYSTEM_PROMPT = (
    "You are a retail customer support agent. Use the tools to answer questions about "
    "products, offers, orders, and policies. When answering from tool results, include the "
    "relevant details such as prices and order numbers. Only answer policy questions from the "
    "search_policies tool result, keeping its [S1] citation markers verbatim; never state "
    "policy from your own knowledge. If an order is not found, state that you could not find "
    "the order. When an order is cancelled, state that the order has been cancelled. "
    "To cancel an order, call cancel_order; the system will ask the customer to confirm. "
    "Be concise."
)


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    customer_id: str
    steps: Annotated[list[str], add]
    citations: tuple[Citation, ...]
    iterations: int
    response: str
    pending_cancellations: list[PendingCancellation]
    iteration_limit: bool


def build_agent_graph(
    *, model: Any, registry: ToolRegistry, max_iterations: int = 6
) -> CompiledStateGraph[Any, Any, Any, Any]:
    tool_defs = getattr(registry, "tools", None) or TOOL_SCHEMAS
    bound_model = model.bind_tools(tool_defs) if hasattr(model, "bind_tools") else model

    def intake(state: AgentState) -> dict[str, Any]:
        has_system = any(
            isinstance(m, SystemMessage) or (isinstance(m, dict) and m.get("role") == "system")
            for m in state.get("messages", [])
        )
        updates: dict[str, Any] = {
            "iterations": 0,
            "steps": Overwrite([]),
            "citations": (),
            "response": "",
            "pending_cancellations": [],
            "iteration_limit": False,
        }
        if not has_system:
            updates["messages"] = [SystemMessage(content=SYSTEM_PROMPT)]
        return updates

    def agent(state: AgentState) -> dict[str, Any]:
        msgs = list(state["messages"])
        system_msgs = [m for m in msgs if isinstance(m, SystemMessage)]
        other_msgs = [m for m in msgs if not isinstance(m, SystemMessage)]
        ordered = system_msgs + other_msgs
        if hasattr(bound_model, "invoke"):
            message = bound_model.invoke(ordered)
        elif hasattr(bound_model, "chat_with_tools"):
            raw = bound_model.chat_with_tools(messages=ordered, tools=tool_defs)
            if isinstance(raw, dict):
                calls = [
                    {
                        "name": c["name"],
                        "args": c.get("arguments") or c.get("args") or {},
                        "id": c.get("id") or f"call_{c['name']}",
                    }
                    for c in raw.get("tool_calls", [])
                ]
                message = AIMessage(content=raw.get("content", ""), tool_calls=calls)
            else:
                message = raw
        else:
            message = bound_model(ordered)
        return {"messages": [message], "iterations": state.get("iterations", 0) + 1}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        raw_calls = (
            getattr(last, "tool_calls", None)
            or (last.get("tool_calls") if isinstance(last, dict) else None)
            or []
        )
        if not raw_calls or state.get("iterations", 0) >= max_iterations:
            return "respond"
        return "tools"

    def tools(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        raw_calls = (
            getattr(last, "tool_calls", None)
            or (last.get("tool_calls") if isinstance(last, dict) else None)
            or []
        )
        results: list[ToolMessage] = []
        steps: list[str] = []
        citations: tuple[Citation, ...] = state.get("citations", ())
        pending_cancellations: list[PendingCancellation] = list(
            state.get("pending_cancellations", [])
        )
        seen_cancellation_orders: set[str] = set()

        for call in raw_calls:
            name = call.get("name", "")
            args = call.get("args") or call.get("arguments") or {}
            call_id = call.get("id") or f"call_{name}"

            if name == "cancel_order":
                order_id = str(args.get("order_id", ""))
                if order_id in seen_cancellation_orders:
                    results.append(
                        ToolMessage(
                            content=(
                                f"Duplicate cancellation request for order {order_id} "
                                "in the same turn."
                            ),
                            name=name,
                            tool_call_id=call_id,
                        )
                    )
                    steps.append(name)
                    continue
                seen_cancellation_orders.add(order_id)
                preview = registry.build_cancellation(order_id, state["customer_id"])
                if isinstance(preview, str):
                    results.append(
                        ToolMessage(content=preview, name=name, tool_call_id=call_id)
                    )
                    steps.append(name)
                else:
                    pending_cancellations.append(
                        PendingCancellation(tool_call_id=call_id, preview=preview)
                    )
            else:
                outcome = registry.run(name, args, customer_id=state["customer_id"])
                content = outcome.content
                if outcome.citations:
                    start_idx = len(citations)
                    renumbered: list[Citation] = []
                    for i, c in enumerate(outcome.citations, start=start_idx + 1):
                        new_marker = f"[S{i}]"
                        if c.marker != new_marker:
                            content = content.replace(c.marker, new_marker)
                        renumbered.append(c.model_copy(update={"marker": new_marker}))
                    citations = citations + tuple(renumbered)
                results.append(ToolMessage(content=content, name=name, tool_call_id=call_id))
                steps.append(name)

        return {
            "messages": results,
            "steps": steps,
            "citations": citations,
            "pending_cancellations": pending_cancellations,
        }

    def route_after_tools(state: AgentState) -> str:
        if state.get("pending_cancellations"):
            return "confirm"
        return "agent"

    def confirm(state: AgentState) -> dict[str, Any]:
        pending = list(state.get("pending_cancellations", []))
        if not pending:
            return {}
        item = pending.pop(0)
        preview = item.preview
        call_id = item.tool_call_id

        decision = interrupt({"order_id": preview.order_id, "summary": preview.summary})
        content = (
            registry.commit_cancellation(preview)
            if str(decision).strip().lower() in {"yes", "y", "confirm"}
            else f"Order {preview.order_id} was not cancelled."
        )
        return {
            "messages": [ToolMessage(content=content, name="cancel_order", tool_call_id=call_id)],
            "steps": ["cancel_order"],
            "pending_cancellations": pending,
        }

    def route_after_confirm(state: AgentState) -> str:
        if state.get("pending_cancellations"):
            return "confirm"
        return "agent"

    def respond(state: AgentState) -> dict[str, Any]:
        msgs = state.get("messages", [])
        last_user_idx = -1
        for i, m in enumerate(msgs):
            if isinstance(m, HumanMessage) or (isinstance(m, dict) and m.get("role") == "user"):
                last_user_idx = i
        current_msgs = msgs[last_user_idx + 1 :] if last_user_idx >= 0 else msgs

        # Check for unfulfilled tool calls if max iterations was hit
        extra_messages: list[ToolMessage] = []
        is_limit = state.get("iterations", 0) >= max_iterations
        if current_msgs:
            last_msg = current_msgs[-1]
            raw_calls = getattr(last_msg, "tool_calls", None) or []
            if raw_calls:
                completed_ids = {
                    getattr(m, "tool_call_id", None)
                    for m in current_msgs
                    if isinstance(m, ToolMessage)
                }
                for c in raw_calls:
                    cid = c.get("id")
                    if cid not in completed_ids:
                        extra_messages.append(
                            ToolMessage(
                                content="Operation cancelled due to iteration limit.",
                                name=c.get("name", "tool"),
                                tool_call_id=cid,
                            )
                        )

        answers = [
            m.content
            for m in current_msgs
            if (isinstance(m, AIMessage) and m.content)
            or (isinstance(m, dict) and m.get("role") == "assistant" and m.get("content"))
        ]
        text = (
            str(answers[-1])
            if answers and answers[-1]
            else "I'm not sure how to help with that."
        )

        # Validate citations against active citation set
        available = state.get("citations", ())
        used_markers = set(re.findall(r"\[S\d+\]", text))
        valid_markers = {c.marker for c in available}
        unknown_markers = used_markers - valid_markers

        if unknown_markers:
            text = (
                "I cannot verify all policy citations in this response. "
                "Please refer to our published policies."
            )
            filtered_citations: tuple[Citation, ...] = ()
        else:
            filtered_citations = tuple(c for c in available if c.marker in used_markers)

        result: dict[str, Any] = {
            "response": text,
            "citations": filtered_citations,
            "iteration_limit": is_limit,
        }
        if extra_messages:
            result["messages"] = extra_messages
        return result

    builder: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
    for name, fn in [
        ("intake", intake),
        ("agent", agent),
        ("tools", tools),
        ("confirm", confirm),
        ("respond", respond),
    ]:
        builder.add_node(name, fn)
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "agent")
    builder.add_conditional_edges("agent", route)
    builder.add_conditional_edges("tools", route_after_tools)
    builder.add_conditional_edges("confirm", route_after_confirm)
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=MemorySaver())


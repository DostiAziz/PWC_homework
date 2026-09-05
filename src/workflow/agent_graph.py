from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from domain.models import Citation
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
        if has_system:
            return {}
        return {"messages": [SystemMessage(content=SYSTEM_PROMPT)]}

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
        if any(c.get("name") == "cancel_order" for c in raw_calls):
            return "confirm"
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
        for call in raw_calls:
            name = call.get("name", "")
            args = call.get("args") or call.get("arguments") or {}
            call_id = call.get("id") or f"call_{name}"
            outcome = registry.run(name, args, customer_id=state["customer_id"])
            results.append(ToolMessage(content=outcome.content, name=name, tool_call_id=call_id))
            steps.append(name)
            if outcome.citations:
                citations = outcome.citations
        return {"messages": results, "steps": steps, "citations": citations}

    def confirm(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        raw_calls = (
            getattr(last, "tool_calls", None)
            or (last.get("tool_calls") if isinstance(last, dict) else None)
            or []
        )
        call = next(c for c in raw_calls if c.get("name") == "cancel_order")
        args = call.get("args") or call.get("arguments") or {}
        call_id = call.get("id") or "call_cancel_order"
        preview = registry.build_cancellation(
            str(args.get("order_id", "")), state["customer_id"]
        )
        if isinstance(preview, str):
            content = preview
        else:
            decision = interrupt({"order_id": preview.order_id, "summary": preview.summary})
            content = (
                registry.commit_cancellation(preview)
                if str(decision).strip().lower() in {"yes", "y", "confirm"}
                else f"Order {preview.order_id} was not cancelled."
            )
        return {
            "messages": [ToolMessage(content=content, name="cancel_order", tool_call_id=call_id)],
            "steps": ["cancel_order"],
        }

    def respond(state: AgentState) -> dict[str, Any]:
        answers = [
            m.content
            for m in state["messages"]
            if (isinstance(m, AIMessage) and m.content)
            or (isinstance(m, dict) and m.get("role") == "assistant" and m.get("content"))
        ]
        text = (
            str(answers[-1])
            if answers and answers[-1]
            else "I'm not sure how to help with that."
        )
        return {"response": text, "citations": state.get("citations", ())}

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
    builder.add_edge("tools", "agent")
    builder.add_edge("confirm", "agent")
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=MemorySaver())

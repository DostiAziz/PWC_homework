from __future__ import annotations

from operator import add
from typing import Annotated, Any, Protocol, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from pwc_support.domain.models import Citation
from pwc_support.workflow.tools import TOOL_SCHEMAS, ToolRegistry

SYSTEM_PROMPT = (
    "You are a retail customer support agent. Use the tools to answer questions about "
    "products, offers, orders, and policies. Only answer policy questions from the "
    "search_policies tool result, keeping its [S1] citation markers verbatim; never state "
    "policy from your own knowledge. To cancel an order, call cancel_order; the system will "
    "ask the customer to confirm. Be concise."
)


class ToolModel(Protocol):
    def chat_with_tools(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]: ...


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict[str, Any]], add]
    customer_id: str
    steps: Annotated[list[str], add]
    citations: tuple[Citation, ...]
    iterations: int
    response: str


def build_agent_graph(
    *, model: ToolModel, registry: ToolRegistry, max_iterations: int = 6
) -> CompiledStateGraph[Any, Any, Any, Any]:
    def intake(state: AgentState) -> dict[str, Any]:
        has_system = any(m.get("role") == "system" for m in state.get("messages", []))
        if has_system:
            return {}
        return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}]}

    def agent(state: AgentState) -> dict[str, Any]:
        msgs = list(state["messages"])
        system_msgs = [m for m in msgs if m.get("role") == "system"]
        other_msgs = [m for m in msgs if m.get("role") != "system"]
        ordered = system_msgs + other_msgs
        message = model.chat_with_tools(messages=ordered, tools=TOOL_SCHEMAS)
        return {"messages": [message], "iterations": state.get("iterations", 0) + 1}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        calls = last.get("tool_calls") or []
        if not calls or state.get("iterations", 0) >= max_iterations:
            return "respond"
        if any(c["name"] == "cancel_order" for c in calls):
            return "confirm"
        return "tools"

    def tools(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        results: list[dict[str, Any]] = []
        steps: list[str] = []
        citations: tuple[Citation, ...] = state.get("citations", ())
        for call in last.get("tool_calls") or []:
            outcome = registry.run(
                call["name"], call["arguments"], customer_id=state["customer_id"]
            )
            results.append({"role": "tool", "tool_name": call["name"], "content": outcome.content})
            steps.append(call["name"])
            if outcome.citations:
                citations = outcome.citations
        return {"messages": results, "steps": steps, "citations": citations}

    def confirm(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        call = next(c for c in last["tool_calls"] if c["name"] == "cancel_order")
        preview = registry.build_cancellation(
            str(call["arguments"].get("order_id", "")), state["customer_id"]
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
            "messages": [{"role": "tool", "tool_name": "cancel_order", "content": content}],
            "steps": ["cancel_order"],
        }

    def respond(state: AgentState) -> dict[str, Any]:
        answers = [m.get("content", "") for m in state["messages"] if m.get("role") == "assistant"]
        text = answers[-1] if answers and answers[-1] else "I'm not sure how to help with that."
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

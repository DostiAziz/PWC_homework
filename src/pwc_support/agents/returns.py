from typing import Protocol, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from pwc_support.agents.contracts import (
    ConversationMemory,
    RoutingTask,
    SpecialistName,
    SpecialistResult,
    SpecialistStatus,
)


class ReturnToolbox(Protocol):
    pass


class ReturnAgentState(TypedDict, total=False):
    task: RoutingTask
    memory: ConversationMemory
    specialist_result: SpecialistResult | None
    memory_update: ConversationMemory | None


def build_return_refund_agent(tools: ReturnToolbox) -> CompiledStateGraph:
    def entry_node(state: ReturnAgentState) -> dict:
        memory = state["memory"]
        task = state["task"]
        
        result = SpecialistResult(
            specialist=SpecialistName.RETURN_REFUND,
            task_id=task.task_id,
            status=SpecialistStatus.CORRECTION_REQUESTED,
            customer_message="ReturnAgent dummy response",
        )
        return {"specialist_result": result, "memory_update": memory}

    builder = StateGraph(ReturnAgentState)
    builder.add_node("entry", entry_node)
    builder.set_entry_point("entry")
    builder.add_edge("entry", END)
    
    return builder.compile()

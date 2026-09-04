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


class ProductToolbox(Protocol):
    pass


class ProductPlanner(Protocol):
    pass


class ProductAgentState(TypedDict, total=False):
    task: RoutingTask
    memory: ConversationMemory
    specialist_result: SpecialistResult | None
    memory_update: ConversationMemory | None


def build_product_agent(tools: ProductToolbox, planner: ProductPlanner) -> CompiledStateGraph:
    def entry_node(state: ProductAgentState) -> dict:
        memory = state["memory"]
        task = state["task"]
        
        result = SpecialistResult(
            specialist=SpecialistName.PRODUCT,
            task_id=task.task_id,
            status=SpecialistStatus.CORRECTION_REQUESTED,
            customer_message="ProductAgent dummy response",
        )
        return {"specialist_result": result, "memory_update": memory}

    builder = StateGraph(ProductAgentState)
    builder.add_node("entry", entry_node)
    builder.set_entry_point("entry")
    builder.add_edge("entry", END)
    
    return builder.compile()

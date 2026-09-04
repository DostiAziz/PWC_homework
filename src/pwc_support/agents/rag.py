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


class RagAnswerer(Protocol):
    pass


class RagAgentState(TypedDict, total=False):
    task: RoutingTask
    memory: ConversationMemory
    specialist_result: SpecialistResult | None
    memory_update: ConversationMemory | None


def build_rag_agent(rag_answerer: RagAnswerer) -> CompiledStateGraph:
    def entry_node(state: RagAgentState) -> dict:
        memory = state["memory"]
        task = state["task"]
        
        result = SpecialistResult(
            specialist=SpecialistName.RAG,
            task_id=task.task_id,
            status=SpecialistStatus.CORRECTION_REQUESTED,
            customer_message="RagAgent dummy response",
        )
        return {"specialist_result": result, "memory_update": memory}

    builder = StateGraph(RagAgentState)
    builder.add_node("entry", entry_node)
    builder.set_entry_point("entry")
    builder.add_edge("entry", END)
    
    return builder.compile()

from typing import Any, Protocol, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from pwc_support.agents.contracts import (
    CancellationReviewPacket,
    ConversationMemory,
    OrderInvestigation,
    PolicyFinding,
    RoutingTask,
    SpecialistName,
    SpecialistResult,
    SpecialistStatus,
)


class OrderToolbox(Protocol):
    def lookup_order(self, order_id: str, customer_id: str) -> dict[str, Any] | None: ...
    def inspect_order(self, order_id: str, customer_id: str) -> dict[str, Any] | None: ...


class OrderAgentState(TypedDict, total=False):
    task: RoutingTask
    memory: ConversationMemory
    customer_id: str
    request: str
    order_id: str | None
    phase: str
    lookup_attempts: int
    messages: list[str]
    facts: dict[str, Any]
    policy_findings: list[PolicyFinding]
    specialist_result: SpecialistResult | None
    memory_update: ConversationMemory | None
    """
    State dict for OrderAgent containing:
    task: RoutingTask
    memory: ConversationMemory
    customer_id: str
    request: str
    order_id: str | None
    phase: str
    lookup_attempts: int
    messages: list[str]
    specialist_result: SpecialistResult | None
    memory_update: ConversationMemory | None
    """
    pass


class OrderNextStep(BaseModel):
    next_node: str


class OrderPlanner(Protocol):
    def next_step(self, state: OrderAgentState) -> OrderNextStep: ...


class OllamaOrderPlanner(OrderPlanner):
    # Stub implementation. Real one would use OllamaGenerator.
    def next_step(self, state: OrderAgentState) -> OrderNextStep:
        pass


def build_order_agent(*, tools: OrderToolbox, planner: OrderPlanner, max_steps: int) -> CompiledStateGraph:
    def merge_memory(state: OrderAgentState) -> dict:
        return {}

    def understand_request(state: OrderAgentState) -> dict:
        if not state.get("order_id"):
            return {"phase": "awaiting_order_id"}
        return {"phase": "lookup_ready"}

    def collect_order_id(state: OrderAgentState) -> dict:
        # returns NEEDS_INFORMATION
        task = state["task"]
        memory = state["memory"]
        msg = "Please provide your order number so I can check it."
        result = SpecialistResult(
            specialist=SpecialistName.ORDER,
            task_id=task.task_id,
            status=SpecialistStatus.NEEDS_INFORMATION,
            customer_message=msg,
        )
        mem_update = memory.model_copy(update={
            "state_json": {"phase": "awaiting_order_id", "lookup_attempts": state.get("lookup_attempts", 0)}
        })
        return {"specialist_result": result, "memory_update": mem_update}

    def lookup_order(state: OrderAgentState) -> dict:
        order_id = state["order_id"]
        customer_id = state["customer_id"]
        found = tools.lookup_order(order_id, customer_id)
        attempts = state.get("lookup_attempts", 0) + 1
        
        if not found:
            if attempts == 1:
                task = state["task"]
                memory = state["memory"]
                result = SpecialistResult(
                    specialist=SpecialistName.ORDER,
                    task_id=task.task_id,
                    status=SpecialistStatus.CORRECTION_REQUESTED,
                    customer_message="I couldn't find that order. Please check the number.",
                )
                mem_update = memory.model_copy(update={
                    "state_json": {"phase": "awaiting_correction", "lookup_attempts": attempts}
                })
                return {"specialist_result": result, "memory_update": mem_update, "lookup_attempts": attempts, "phase": "awaiting_correction"}
            else:
                task = state["task"]
                memory = state["memory"]
                result = SpecialistResult(
                    specialist=SpecialistName.ORDER,
                    task_id=task.task_id,
                    status=SpecialistStatus.ORDER_NOT_FOUND,
                    customer_message="I still can't find that order.",
                )
                mem_update = memory.model_copy(update={
                    "state_json": {"phase": "new", "lookup_attempts": 0},
                    "active_specialist": None,
                    "active_task_id": None
                })
                return {"specialist_result": result, "memory_update": mem_update, "lookup_attempts": attempts, "phase": "not_found"}
        
        return {"phase": "inspect_ready", "lookup_attempts": attempts}

    def inspect_order(state: OrderAgentState) -> dict:
        order_id = state["order_id"]
        customer_id = state["customer_id"]
        facts = tools.inspect_order(order_id, customer_id)
        return {"phase": "evaluate_policy", "facts": facts}

    def evaluate_cancellation_policy(state: OrderAgentState) -> dict:
        facts = state["facts"]
        findings = []
        if facts.get("status") == "cancelled":
            findings.append(PolicyFinding(rule_id="already_cancelled", passed=False, message="Order is already cancelled"))
        elif facts.get("shipped_at") or facts.get("fulfilment_status") == "shipped":
            findings.append(PolicyFinding(rule_id="already_shipped", passed=False, message="Order has already shipped"))
        else:
            findings.append(PolicyFinding(rule_id="not_shipped", passed=True, message="Order has not shipped"))
        
        return {"policy_findings": findings, "phase": "prepare_review"}

    def prepare_cancellation_review(state: OrderAgentState) -> dict:
        facts = state["facts"]
        findings = state["policy_findings"]
        task = state["task"]
        memory = state["memory"]
        
        investigation = OrderInvestigation(**facts)
        packet = CancellationReviewPacket(
            conversation_id=memory.conversation_id,
            client_id=memory.client_id,
            original_request=state["request"],
            order=investigation,
            policy_findings=findings,
            recommended_action="human_review"
        )
        
        result = SpecialistResult(
            specialist=SpecialistName.ORDER,
            task_id=task.task_id,
            status=SpecialistStatus.REVIEW_REQUIRED,
            review_packet=packet,
        )
        mem_update = memory.model_copy(update={
            "state_json": {"phase": "new", "lookup_attempts": 0},
            "active_specialist": None,
            "active_task_id": None
        })
        return {"specialist_result": result, "memory_update": mem_update, "phase": "done"}

    def return_to_main_workflow(state: OrderAgentState) -> dict:
        return {}

    def next_step_router(state: OrderAgentState) -> str:
        if state.get("specialist_result"):
            return "return_to_main_workflow"
            
        phase = state.get("phase", "new")
        if phase in ("awaiting_order_id", "awaiting_correction"):
            if not state.get("order_id"):
                return "collect_order_id"
            else:
                return "lookup_order"
        elif phase == "new":
            return "understand_request"
        elif phase == "lookup_ready":
            return "lookup_order"
        elif phase == "inspect_ready":
            return "inspect_order"
        elif phase == "evaluate_policy":
            return "evaluate_cancellation_policy"
        elif phase == "prepare_review":
            return "prepare_cancellation_review"
        return "return_to_main_workflow"

    builder = StateGraph(OrderAgentState)
    builder.add_node("merge_memory", merge_memory)
    builder.add_node("understand_request", understand_request)
    builder.add_node("collect_order_id", collect_order_id)
    builder.add_node("lookup_order", lookup_order)
    builder.add_node("inspect_order", inspect_order)
    builder.add_node("evaluate_cancellation_policy", evaluate_cancellation_policy)
    builder.add_node("prepare_cancellation_review", prepare_cancellation_review)
    builder.add_node("return_to_main_workflow", return_to_main_workflow)

    builder.set_entry_point("merge_memory")
    builder.add_conditional_edges("merge_memory", next_step_router)
    builder.add_conditional_edges("understand_request", next_step_router)
    builder.add_conditional_edges("collect_order_id", next_step_router)
    builder.add_conditional_edges("lookup_order", next_step_router)
    builder.add_conditional_edges("inspect_order", next_step_router)
    builder.add_conditional_edges("evaluate_cancellation_policy", next_step_router)
    builder.add_conditional_edges("prepare_cancellation_review", next_step_router)
    builder.add_edge("return_to_main_workflow", END)

    return builder.compile()

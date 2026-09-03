from __future__ import annotations

from typing import Annotated, Any, TypedDict

from pwc_support.domain.models import TaskResult
from pwc_support.workflow.reducers import append_events, merge_rag_results, merge_task_results


class SupportState(TypedDict, total=False):
    """Shared workflow state: every node reads and writes named slices of it."""

    message: dict[str, Any]
    run_id: str
    conversation_id: str
    thread_id: str
    client_id: str
    channel: str
    case_id: str | None
    triage: dict[str, Any]
    semantic_risk: dict[str, Any]
    routing_snapshot: dict[str, Any]
    final_routing: dict[str, Any]
    inbound_claim: dict[str, Any]
    evidence: dict[str, Any]
    plan: dict[str, Any]
    expected_task_ids: list[str]
    task: dict[str, Any]
    task_results: Annotated[dict[str, TaskResult], merge_task_results]
    rag_results: Annotated[list[dict[str, Any]], merge_rag_results]
    draft: dict[str, Any]
    draft_version: int
    proposed_actions: list[dict[str, Any]]
    tool_calls: Annotated[list[dict[str, Any]], append_events]
    verification: dict[str, Any]
    review_request: dict[str, Any]
    review_decision: dict[str, Any]
    escalation_intent: dict[str, Any]
    inbound_claim_result: dict[str, Any]
    review_decision_result: dict[str, Any]
    events: Annotated[list[dict[str, Any]], append_events]
    delivery: dict[str, Any]
    error: dict[str, Any]
    outcome: dict[str, Any]
    retail_intent: str
    retail_answer: str
    tool_results: dict[str, Any]
    action_proposal: dict[str, Any]
    action_risk: str
    return_id: str

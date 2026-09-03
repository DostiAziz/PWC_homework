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
    events: Annotated[list[dict[str, Any]], append_events]
    delivery: dict[str, Any]
    error: dict[str, Any]
    outcome: dict[str, Any]

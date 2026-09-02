from __future__ import annotations

from typing import Annotated, Any, TypedDict

from pwc_support.domain.models import TaskResult
from pwc_support.workflow.reducers import merge_task_results


class SupportState(TypedDict, total=False):
    message: dict[str, Any]
    run_id: str
    duplicate: bool
    case_id: str | None
    triage: dict[str, Any]
    plan: dict[str, Any]
    expected_task_ids: list[str]
    task_results: Annotated[dict[str, TaskResult], merge_task_results]
    rag_results: list[dict[str, Any]]
    draft: dict[str, Any]
    draft_version: int
    proposed_actions: list[dict[str, Any]]
    verification: dict[str, Any]
    review_request: dict[str, Any]
    review_decision: dict[str, Any]
    revision_count: int
    events: list[dict[str, Any]]
    delivery: dict[str, Any]
    error: dict[str, Any]
    outcome: dict[str, Any]

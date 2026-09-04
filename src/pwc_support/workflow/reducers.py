from __future__ import annotations

from typing import Any

from pwc_support.domain.models import SpecialistResult, TaskResult


def merge_task_results(
    left: dict[str, TaskResult] | None,
    right: dict[str, TaskResult] | None,
) -> dict[str, TaskResult]:
    merged = dict(left or {})
    for task_id, result in (right or {}).items():
        existing = merged.get(task_id)
        if existing is not None and existing != result:
            raise ValueError(f"conflicting task result: {task_id}")
        merged[task_id] = result
    return merged


def merge_specialist_results(
    left: dict[str, SpecialistResult] | None,
    right: dict[str, SpecialistResult] | None,
) -> dict[str, SpecialistResult]:
    """Collect one specialist result per routed task, rejecting conflicting replays."""
    merged = dict(left or {})
    for task_id, result in (right or {}).items():
        existing = merged.get(task_id)
        if existing is not None and existing != result:
            raise ValueError(f"conflicting specialist result: {task_id}")
        merged[task_id] = result
    return merged


def append_events(
    left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Append-only reducer so fan-out branches never overwrite each other's trace."""
    return [*(left or []), *(right or [])]


def merge_rag_results(
    left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Collect one RAG result per knowledge task, keyed by task id and order-stable."""
    merged = list(left or [])
    known = {item.get("task_id") for item in merged}
    for item in right or []:
        if item.get("task_id") in known:
            continue
        known.add(item.get("task_id"))
        merged.append(item)
    return sorted(merged, key=lambda item: str(item.get("task_id", "")))

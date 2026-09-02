from pwc_support.domain.models import TaskResult


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

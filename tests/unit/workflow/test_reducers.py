import pytest

from pwc_support.domain.models import TaskResult, TaskStatus
from pwc_support.workflow.reducers import merge_task_results


def test_reducer_rejects_incompatible_duplicate() -> None:
    first = TaskResult(task_id="task-1", status=TaskStatus.SUCCESS, payload={"a": 1})
    second = TaskResult(task_id="task-1", status=TaskStatus.SUCCESS, payload={"a": 2})
    with pytest.raises(ValueError, match="conflicting task result"):
        merge_task_results({"task-1": first}, {"task-1": second})

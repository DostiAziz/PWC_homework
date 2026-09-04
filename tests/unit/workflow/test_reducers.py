import pytest

from pwc_support.domain.models import (
    SpecialistName,
    SpecialistResult,
    SpecialistStatus,
    TaskResult,
    TaskStatus,
)
from pwc_support.workflow.reducers import merge_specialist_results, merge_task_results


def test_reducer_rejects_incompatible_duplicate() -> None:
    first = TaskResult(task_id="task-1", status=TaskStatus.SUCCESS, payload={"a": 1})
    second = TaskResult(task_id="task-1", status=TaskStatus.SUCCESS, payload={"a": 2})
    with pytest.raises(ValueError, match="conflicting task result"):
        merge_task_results({"task-1": first}, {"task-1": second})


def test_reducer_rejects_incompatible_specialist_duplicate() -> None:
    first = SpecialistResult(
        specialist=SpecialistName.ORDER,
        task_id="order-1",
        status=SpecialistStatus.REVIEW_REQUIRED,
        facts={"order_id": "ORD-1001"},
    )
    second = SpecialistResult(
        specialist=SpecialistName.ORDER,
        task_id="order-1",
        status=SpecialistStatus.COMPLETED,
        facts={"order_id": "ORD-1001"},
    )
    with pytest.raises(ValueError, match="conflicting specialist result"):
        merge_specialist_results({"order-1": first}, {"order-1": second})


def test_reducer_merges_specialist_results_by_task_id() -> None:
    order_result = SpecialistResult(
        specialist=SpecialistName.ORDER,
        task_id="order-1",
        status=SpecialistStatus.COMPLETED,
    )
    rag_result = SpecialistResult(
        specialist=SpecialistName.RAG,
        task_id="rag-1",
        status=SpecialistStatus.COMPLETED,
    )

    merged = merge_specialist_results({"order-1": order_result}, {"rag-1": rag_result})

    assert merged == {"order-1": order_result, "rag-1": rag_result}

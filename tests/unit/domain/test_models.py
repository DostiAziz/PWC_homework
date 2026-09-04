import pytest
from pydantic import ValidationError

from pwc_support.domain.models import (
    CatalogueAction,
    OrderAction,
    Task,
    TaskKind,
)


def test_catalogue_task_rejects_order_action() -> None:
    with pytest.raises(ValidationError):
        Task(
            task_id="task-1",
            kind=TaskKind.CATALOGUE,
            request="Show offers",
            catalogue_action=CatalogueAction.OFFERS,
            order_action=OrderAction.CANCEL,
        )


def test_order_task_normalizes_order_id() -> None:
    task = Task(
        task_id="task-1",
        kind=TaskKind.ORDER,
        request="Cancel ord-2001",
        order_action=OrderAction.CANCEL,
        order_id="ord-2001",
    )

    assert task.order_id == "ORD-2001"

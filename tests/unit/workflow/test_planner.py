from typing import Any

import pytest

from pwc_support.domain.models import CatalogueAction, OrderAction, TaskKind
from pwc_support.workflow.planner import (
    OllamaPlanner,
    PlanningUnavailable,
    is_greeting,
    parse_confirmation,
)


class FakeStructuredModel:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def structured(self, *, schema: type, **_: Any) -> Any:
        return schema.model_validate(self.payload)


class MalformedStructuredModel:
    def structured(self, **_: Any) -> None:
        return None


class UnavailableStructuredModel:
    def structured(self, **_: Any) -> None:
        raise ConnectionError("Ollama is unavailable")


def test_planner_decomposes_compound_request_once_per_kind() -> None:
    model = FakeStructuredModel(
        {
            "tasks": [
                {
                    "kind": "catalogue",
                    "request": "show jacket offers",
                    "product_query": "jacket",
                    "catalogue_action": "offers",
                },
                {
                    "kind": "order",
                    "request": "cancel ORD-2001",
                    "order_id": "ORD-2001",
                    "order_action": "cancel",
                },
            ]
        }
    )

    tasks = OllamaPlanner(model).plan("Show jacket offers and cancel ORD-2001")

    assert [task.task_id for task in tasks] == ["task-1", "task-2"]
    assert [task.kind for task in tasks] == [TaskKind.CATALOGUE, TaskKind.ORDER]
    assert tasks[0].catalogue_action is CatalogueAction.OFFERS
    assert tasks[1].order_action is OrderAction.CANCEL


def test_planner_rejects_duplicate_task_kinds() -> None:
    model = FakeStructuredModel(
        {
            "tasks": [
                {"kind": "knowledge", "request": "shipping"},
                {"kind": "knowledge", "request": "warranty"},
            ]
        }
    )

    with pytest.raises(PlanningUnavailable, match="one task per kind"):
        OllamaPlanner(model).plan("Tell me about shipping and warranty")


def test_planner_normalizes_malformed_gateway_output() -> None:
    with pytest.raises(PlanningUnavailable, match="invalid task plan"):
        OllamaPlanner(MalformedStructuredModel()).plan("Show offers")


def test_planner_normalizes_unavailable_gateway() -> None:
    with pytest.raises(PlanningUnavailable, match="planner is unavailable"):
        OllamaPlanner(UnavailableStructuredModel()).plan("Show offers")


def test_greeting_and_confirmation_are_deterministic() -> None:
    assert is_greeting("Hi")
    assert parse_confirmation("yes, cancel it") == "yes"
    assert parse_confirmation("no") == "no"
    assert parse_confirmation("maybe") == "unclear"

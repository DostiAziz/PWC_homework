from __future__ import annotations

import re
from typing import Any, Protocol

from pydantic import Field, ValidationError

from pwc_support.domain.models import (
    CatalogueAction,
    DomainModel,
    OrderAction,
    Task,
    TaskKind,
)

SYSTEM_PROMPT = """Classify a retail support message into at most one task per kind.
Allowed kinds: knowledge, catalogue, order.
Catalogue actions: search, offers. Order actions: lookup, cancel.
Combine requests of the same kind into one task. Never invent an order ID.
Return only the requested schema."""

GREETING = re.compile(r"^(hi|hello|hey|good (morning|afternoon|evening))[!. ]*$", re.I)
YES = re.compile(r"^(yes|y|confirm|yes,? cancel it|cancel it)[!. ]*$", re.I)
NO = re.compile(
    r"^(no|n|stop|no,?\s*(do not cancel|don't cancel)|do not cancel|don't cancel)[!. ]*$",
    re.I,
)


class PlanningUnavailable(RuntimeError):
    pass


class StructuredModel(Protocol):
    def structured(
        self, *, system: str, user: str, schema: type, temperature: float = 0.0
    ) -> Any: ...


class _TaskDraft(DomainModel):
    kind: TaskKind
    request: str = Field(min_length=1, max_length=1000)
    product_query: str | None = Field(default=None, max_length=200)
    order_id: str | None = Field(default=None, max_length=64)
    catalogue_action: CatalogueAction | None = None
    order_action: OrderAction | None = None


class _Plan(DomainModel):
    tasks: tuple[_TaskDraft, ...] = Field(min_length=1, max_length=3)


class Planner(Protocol):
    def plan(self, message: str) -> tuple[Task, ...]: ...


class OllamaPlanner:
    def __init__(self, model: StructuredModel) -> None:
        self.model = model

    def plan(self, message: str) -> tuple[Task, ...]:
        try:
            output = self.model.structured(
                system=SYSTEM_PROMPT,
                user=f"<untrusted_message>{message}</untrusted_message>",
                schema=_Plan,
                temperature=0.0,
            )
            if not isinstance(output, _Plan):
                raise TypeError("planner returned an unexpected output type")
            kinds = [draft.kind for draft in output.tasks]
            if len(kinds) != len(set(kinds)):
                raise PlanningUnavailable("planner must return one task per kind")
            return tuple(
                Task.model_validate({"task_id": f"task-{index}", **draft.model_dump()})
                for index, draft in enumerate(output.tasks, start=1)
            )
        except PlanningUnavailable:
            raise
        except (ValidationError, TypeError, ValueError) as error:
            raise PlanningUnavailable("planner returned an invalid task plan") from error
        except Exception as error:
            raise PlanningUnavailable("planner is unavailable") from error


def is_greeting(message: str) -> bool:
    return bool(GREETING.fullmatch(message.strip()))


def parse_confirmation(message: str) -> str:
    normalized = message.strip()
    if YES.fullmatch(normalized):
        return "yes"
    if NO.fullmatch(normalized):
        return "no"
    return "unclear"

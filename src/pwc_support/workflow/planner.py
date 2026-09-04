from __future__ import annotations

import re
from typing import Any, Protocol

from pydantic import Field, ValidationError, model_validator

from pwc_support.domain.models import (
    CatalogueAction,
    DomainModel,
    OrderAction,
    Task,
    TaskKind,
)

SYSTEM_PROMPT = """Classify a retail support message into at most one task per kind.
Allowed kinds: knowledge, catalogue, order.
For catalogue tasks: set catalogue_action to "search" or "offers".
For order tasks: set order_action to "lookup" or "cancel", and extract order_id if mentioned.
For knowledge tasks: policy, shipping, warranty, returns; leave actions null.
Combine requests of the same kind into one task. Never invent an order ID.
Return only the requested schema."""

GREETING = re.compile(r"^(hi|hello|hey|good (morning|afternoon|evening))[!. ]*$", re.I)
YES = re.compile(r"^(yes|y|confirm|yes,? cancel it|cancel it)[!. ]*$", re.I)
NO = re.compile(
    r"^(no|n|stop|no,?\s*(do not cancel|don't cancel)|do not cancel|don't cancel)[!. ]*$",
    re.I,
)
ORDER_ID_RE = re.compile(r"\b(ORD-\d+)\b", re.I)


class PlanningUnavailable(RuntimeError):
    pass


class StructuredModel(Protocol):
    def structured(
        self, *, system: str, user: str, schema: type, temperature: float = 0.0
    ) -> Any: ...


def _extract_product_query(text: str) -> str:
    cleaned = re.sub(r"[?!.,'\"]", " ", text)
    stop_words = {
        "what",
        "which",
        "are",
        "is",
        "there",
        "any",
        "available",
        "do",
        "you",
        "have",
        "sell",
        "show",
        "me",
        "find",
        "looking",
        "for",
        "please",
        "can",
        "i",
        "get",
        "see",
        "the",
        "in",
        "stock",
        "offers",
        "offer",
        "discount",
        "discounts",
        "deals",
        "deal",
        "cheap",
        "best",
        "good",
        "some",
        "a",
        "an",
        "products",
        "product",
        "items",
        "item",
        "search",
    }
    words = [w for w in cleaned.split() if w.lower() not in stop_words]
    return " ".join(words).strip()


class _TaskDraft(DomainModel):
    kind: TaskKind
    request: str = Field(min_length=1, max_length=1000)
    product_query: str | None = Field(default=None, max_length=200)
    order_id: str | None = Field(default=None, max_length=64)
    catalogue_action: CatalogueAction | None = None
    order_action: OrderAction | None = None

    @model_validator(mode="after")
    def populate_defaults(self) -> _TaskDraft:
        if self.kind is TaskKind.CATALOGUE:
            if self.catalogue_action is None:
                text = (self.request + " " + (self.product_query or "")).casefold()
                if any(w in text for w in ("offer", "discount", "deal", "sale", "promo")):
                    self.catalogue_action = CatalogueAction.OFFERS
                else:
                    self.catalogue_action = CatalogueAction.SEARCH
            if not self.product_query or self.product_query.casefold() in ("search", "offers"):
                extracted = _extract_product_query(self.request)
                if extracted:
                    self.product_query = extracted
            self.order_id = None
            self.order_action = None
        elif self.kind is TaskKind.ORDER:
            if self.order_action is None:
                text = self.request.casefold()
                if any(w in text for w in ("cancel", "stop", "abort")):
                    self.order_action = OrderAction.CANCEL
                else:
                    self.order_action = OrderAction.LOOKUP
            if self.order_id is None:
                match = ORDER_ID_RE.search(self.request)
                if match:
                    self.order_id = match.group(1).upper()
            self.product_query = None
            self.catalogue_action = None
        elif self.kind is TaskKind.KNOWLEDGE:
            self.product_query = None
            self.order_id = None
            self.catalogue_action = None
            self.order_action = None
        return self


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
            tasks: list[Task] = []
            for index, draft in enumerate(output.tasks, start=1):
                data = draft.model_dump()
                if len(output.tasks) == 1 and (
                    len(draft.request.split()) <= 2
                    or draft.request.casefold()
                    in (
                        "search",
                        "offers",
                        "lookup",
                        "cancel",
                        "shipping",
                        "order",
                        "knowledge",
                        "catalogue",
                    )
                ):
                    data["request"] = message
                    if draft.kind is TaskKind.CATALOGUE and not data.get("product_query"):
                        extracted = _extract_product_query(message)
                        if extracted:
                            data["product_query"] = extracted
                if draft.kind is TaskKind.ORDER:
                    if not data.get("order_id"):
                        match = ORDER_ID_RE.search(data.get("request", "")) or ORDER_ID_RE.search(
                            message
                        )
                        if match:
                            data["order_id"] = match.group(1).upper()
                    if not data.get("order_action"):
                        text = (data.get("request", "") + " " + message).casefold()
                        if any(w in text for w in ("cancel", "stop", "abort")):
                            data["order_action"] = OrderAction.CANCEL
                        else:
                            data["order_action"] = OrderAction.LOOKUP
                tasks.append(Task.model_validate({"task_id": f"task-{index}", **data}))
            return tuple(tasks)
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

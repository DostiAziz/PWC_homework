from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from langgraph.types import Command

from retail_support.domain.models import ChatReply

logger = logging.getLogger(__name__)


class AgentService:
    def __init__(self, graph: Any) -> None:
        self.graph = graph

    def submit(self, *, body: str, customer_id: str, thread_id: str | None = None) -> ChatReply:
        tid = thread_id or str(uuid.uuid4())
        payload = {"messages": [{"role": "user", "content": body}], "customer_id": customer_id}
        return self._run(payload, tid)

    def resume(self, *, thread_id: str, decision: str) -> ChatReply:
        return self._run(Command(resume=decision), thread_id)

    def _run(self, payload: Any, thread_id: str) -> ChatReply:
        started = time.perf_counter()
        config = {"configurable": {"thread_id": thread_id}}
        try:
            terminal = self.graph.invoke(payload, config)
        except Exception:
            logger.exception("agent failed", extra={"thread_id": thread_id})
            return ChatReply(
                message="The support agent is unavailable. Please try again.",
                total_duration_ms=self._ms(started),
            )
        interrupts = terminal.get("__interrupt__")
        if interrupts:
            value = interrupts[0].value
            return ChatReply(
                message=value.get("summary", "") + "? Please answer yes or no.",
                awaiting_confirmation=True,
                preview=value.get("summary", ""),
                steps=("cancel_order",),
                total_duration_ms=self._ms(started),
            )
        return ChatReply(
            message=str(terminal.get("response", "")),
            citations=tuple(terminal.get("citations", ())),
            steps=tuple(terminal.get("steps", ())),
            total_duration_ms=self._ms(started),
        )

    @staticmethod
    def _ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)


ChatService = AgentService

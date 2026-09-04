from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from pwc_support.domain.models import CancellationPreview, ChatReply, TraceEvent

logger = logging.getLogger(__name__)


class SupportGraph(Protocol):
    def invoke(self, input: dict[str, Any], /) -> dict[str, Any] | Any: ...


class ChatService:
    def __init__(self, graph: SupportGraph) -> None:
        self.graph = graph

    def submit(
        self,
        *,
        body: str,
        customer_id: str,
        pending_cancellation: CancellationPreview | None = None,
    ) -> ChatReply:
        started = time.perf_counter()
        try:
            terminal = self.graph.invoke(
                {
                    "message": body,
                    "customer_id": customer_id,
                    "pending_cancellation": pending_cancellation,
                }
            )
        except Exception:
            logger.exception("support workflow failed", extra={"customer_id": customer_id})
            return ChatReply(
                message="The support workflow is unavailable. Please try again.",
                events=(TraceEvent(node="service", event_type="failed"),),
                total_duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        return ChatReply(
            message=str(terminal["response"]),
            citations=tuple(terminal.get("citations", ())),
            tasks=tuple(terminal.get("tasks", ())),
            events=tuple(terminal.get("events", ())),
            pending_cancellation=terminal.get("pending_cancellation"),
            total_duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )

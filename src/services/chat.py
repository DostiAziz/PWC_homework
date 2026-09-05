from __future__ import annotations

import logging
import time
import uuid
from threading import Lock
from typing import Any, Literal

from langgraph.types import Command

from domain.models import ChatReply
from observability import get_timing_spans, record_span, trace_request, traceable

logger = logging.getLogger(__name__)


class AgentService:
    def __init__(self, graph: Any) -> None:
        self.graph = graph
        self._thread_owners: dict[str, str] = {}
        self._awaiting_threads: set[str] = set()
        self._thread_locks: dict[str, Lock] = {}
        self._global_lock = Lock()

    def _get_lock(self, thread_id: str) -> Lock:
        with self._global_lock:
            if thread_id not in self._thread_locks:
                self._thread_locks[thread_id] = Lock()
            return self._thread_locks[thread_id]

    @traceable(name="AgentService.submit", run_type="chain")
    def submit(self, *, body: str, customer_id: str, thread_id: str | None = None) -> ChatReply:
        tid = thread_id or str(uuid.uuid4())
        with self._get_lock(tid):
            owner = self._thread_owners.get(tid)
            if owner is not None and owner != customer_id:
                return ChatReply(
                    message="Access denied: conversation belongs to another customer.",
                    status="invalid_request",
                )
            if tid in self._awaiting_threads:
                return ChatReply(
                    message=(
                        "A confirmation is already pending for this conversation. "
                        "Please answer yes or no."
                    ),
                    awaiting_confirmation=True,
                    status="awaiting_confirmation",
                )
            self._thread_owners[tid] = customer_id
            payload = {"messages": [{"role": "user", "content": body}], "customer_id": customer_id}
            req_id = str(uuid.uuid4())
            with trace_request(req_id, metadata={"customer_id": customer_id, "thread_id": tid}):
                with record_span(
                    "agent.submit",
                    run_type="chain",
                    metadata={"customer_id": customer_id, "thread_id": tid},
                ):
                    reply = self._run(payload, tid, request_id=req_id, customer_id=customer_id)
                return reply.model_copy(update={"request_id": req_id, "spans": get_timing_spans()})

    @traceable(name="AgentService.resume", run_type="chain")
    def resume(self, *, thread_id: str, customer_id: str, decision: str) -> ChatReply:
        with self._get_lock(thread_id):
            owner = self._thread_owners.get(thread_id)
            if owner is None:
                return ChatReply(
                    message="The session has expired or is invalid. Please submit a new request.",
                    status="invalid_request",
                )
            if owner != customer_id:
                return ChatReply(
                    message="Access denied: conversation belongs to another customer.",
                    status="invalid_request",
                )
            req_id = str(uuid.uuid4())
            with trace_request(
                req_id, metadata={"customer_id": customer_id, "thread_id": thread_id}
            ):
                with record_span(
                    "agent.resume",
                    run_type="chain",
                    metadata={"customer_id": customer_id, "thread_id": thread_id},
                ):
                    reply = self._run(
                        Command(resume=decision),
                        thread_id,
                        request_id=req_id,
                        customer_id=customer_id,
                    )
                return reply.model_copy(update={"request_id": req_id, "spans": get_timing_spans()})

    def _run(
        self,
        payload: Any,
        thread_id: str,
        *,
        request_id: str = "",
        customer_id: str = "",
    ) -> ChatReply:
        started = time.perf_counter()
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "metadata": {
                "thread_id": thread_id,
                "customer_id": customer_id,
                "request_id": request_id,
            },
            "tags": ["retail-support"],
            "run_name": "agent.submit" if not isinstance(payload, Command) else "agent.resume",
        }
        try:
            terminal = self.graph.invoke(payload, config)
        except Exception:
            logger.exception("agent failed", extra={"thread_id": thread_id})
            return ChatReply(
                message="The support agent is unavailable. Please try again.",
                status="unavailable",
                total_duration_ms=self._ms(started),
            )
        interrupts = terminal.get("__interrupt__")
        if interrupts:
            self._awaiting_threads.add(thread_id)
            value = interrupts[0].value
            return ChatReply(
                message=value.get("summary", "") + "? Please answer yes or no.",
                awaiting_confirmation=True,
                preview=value.get("summary", ""),
                steps=("cancel_order",),
                status="awaiting_confirmation",
                total_duration_ms=self._ms(started),
            )
        self._awaiting_threads.discard(thread_id)
        status: Literal[
            "answered",
            "awaiting_confirmation",
            "insufficient_evidence",
            "unavailable",
            "iteration_limit",
            "invalid_request",
        ] = "answered"
        if terminal.get("iteration_limit"):
            status = "iteration_limit"
        elif terminal.get("status") == "insufficient_evidence":
            status = "insufficient_evidence"

        return ChatReply(
            message=str(terminal.get("response", "")),
            citations=tuple(terminal.get("citations", ())),
            steps=tuple(terminal.get("steps", ())),
            status=status,
            total_duration_ms=self._ms(started),
        )

    @staticmethod
    def _ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)

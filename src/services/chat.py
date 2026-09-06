from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal

from langgraph.types import Command

from domain.models import ChatReply
from observability import get_timing_spans, record_span, trace_request

logger = logging.getLogger(__name__)


@dataclass
class ChatStreamEvent:
    kind: Literal["step", "token", "interrupt", "done", "error"]
    token: str = ""
    step: str = ""
    reply: ChatReply | None = None


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

    def submit(self, *, body: str, customer_id: str, thread_id: str | None = None) -> ChatReply:
        last_reply: ChatReply | None = None
        for event in self.stream_submit(body=body, customer_id=customer_id, thread_id=thread_id):
            if event.reply is not None:
                last_reply = event.reply
        return last_reply or ChatReply(
            message="Something went wrong while processing your request.",
            status="unavailable",
        )

    def stream_submit(
        self, *, body: str, customer_id: str, thread_id: str | None = None
    ) -> Iterator[ChatStreamEvent]:
        tid = thread_id or str(uuid.uuid4())
        with self._get_lock(tid):
            owner = self._thread_owners.get(tid)
            if owner is not None and owner != customer_id:
                reply = ChatReply(
                    message="Access denied: conversation belongs to another customer.",
                    status="invalid_request",
                )
                yield ChatStreamEvent(kind="error", reply=reply)
                return
            if tid in self._awaiting_threads:
                reply = ChatReply(
                    message=(
                        "A confirmation is already pending for this conversation. "
                        "Please answer yes or no."
                    ),
                    awaiting_confirmation=True,
                    status="awaiting_confirmation",
                )
                yield ChatStreamEvent(kind="interrupt", reply=reply, token=reply.message)
                return
            self._thread_owners[tid] = customer_id
            payload = {"messages": [{"role": "user", "content": body}], "customer_id": customer_id}
            req_id = str(uuid.uuid4())
            with (
                trace_request(req_id, metadata={"customer_id": customer_id, "thread_id": tid}),
                record_span(
                    "agent.submit_stream",
                    run_type="chain",
                    metadata={"customer_id": customer_id, "thread_id": tid},
                ),
            ):
                for event in self._run_stream(
                    payload, tid, request_id=req_id, customer_id=customer_id
                ):
                    if event.reply is not None:
                        event.reply = event.reply.model_copy(
                            update={"request_id": req_id, "spans": get_timing_spans()}
                        )
                    yield event

    def resume(self, *, thread_id: str, customer_id: str, decision: str) -> ChatReply:
        last_reply: ChatReply | None = None
        for event in self.stream_resume(
            thread_id=thread_id, customer_id=customer_id, decision=decision
        ):
            if event.reply is not None:
                last_reply = event.reply
        return last_reply or ChatReply(
            message="Something went wrong while processing your request.",
            status="unavailable",
        )

    def stream_resume(
        self, *, thread_id: str, customer_id: str, decision: str
    ) -> Iterator[ChatStreamEvent]:
        with self._get_lock(thread_id):
            owner = self._thread_owners.get(thread_id)
            if owner is None:
                reply = ChatReply(
                    message="The session has expired or is invalid. Please submit a new request.",
                    status="invalid_request",
                )
                yield ChatStreamEvent(kind="error", reply=reply)
                return
            if owner != customer_id:
                reply = ChatReply(
                    message="Access denied: conversation belongs to another customer.",
                    status="invalid_request",
                )
                yield ChatStreamEvent(kind="error", reply=reply)
                return
            req_id = str(uuid.uuid4())
            with (
                trace_request(
                    req_id, metadata={"customer_id": customer_id, "thread_id": thread_id}
                ),
                record_span(
                    "agent.resume_stream",
                    run_type="chain",
                    metadata={"customer_id": customer_id, "thread_id": thread_id},
                ),
            ):
                for event in self._run_stream(
                    Command(resume=decision),
                    thread_id,
                    request_id=req_id,
                    customer_id=customer_id,
                ):
                    if event.reply is not None:
                        event.reply = event.reply.model_copy(
                            update={"request_id": req_id, "spans": get_timing_spans()}
                        )
                    yield event

    def _run_stream(
        self,
        payload: Any,
        thread_id: str,
        *,
        request_id: str = "",
        customer_id: str = "",
    ) -> Iterator[ChatStreamEvent]:
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

        # Fallback to invoke if graph does not support stream
        if not hasattr(self.graph, "stream"):
            reply = self._run(payload, thread_id, request_id=request_id, customer_id=customer_id)
            if reply.status in ("unavailable", "invalid_request"):
                yield ChatStreamEvent(kind="error", reply=reply)
            elif reply.awaiting_confirmation:
                yield ChatStreamEvent(kind="interrupt", reply=reply, token=reply.message)
            else:
                for step in reply.steps:
                    yield ChatStreamEvent(kind="step", step=step)
                if reply.message:
                    yield ChatStreamEvent(kind="token", token=reply.message)
                yield ChatStreamEvent(kind="done", reply=reply)
            return

        accumulated_steps: list[str] = []
        terminal_respond: dict[str, Any] = {}
        streamed_tokens: list[str] = []

        try:
            for mode, chunk in self.graph.stream(
                payload, config, stream_mode=["messages", "updates"]
            ):
                if mode == "messages":
                    msg, meta = chunk
                    node = meta.get("langgraph_node")
                    if node == "agent":
                        tool_calls = getattr(msg, "tool_calls", None) or getattr(
                            msg, "tool_call_chunks", None
                        )
                        content = getattr(msg, "content", None)
                        if not tool_calls and isinstance(content, str) and content:
                            streamed_tokens.append(content)
                            yield ChatStreamEvent(kind="token", token=content)

                elif mode == "updates":
                    if "__interrupt__" in chunk:
                        self._awaiting_threads.add(thread_id)
                        interrupts = chunk["__interrupt__"]
                        int_obj = interrupts[0] if interrupts else None
                        val = getattr(int_obj, "value", int_obj) if int_obj is not None else {}
                        summary = val.get("summary", "") if isinstance(val, dict) else str(val)
                        reply = ChatReply(
                            message=summary + "? Please answer yes or no.",
                            awaiting_confirmation=True,
                            preview=summary,
                            steps=("cancel_order",),
                            status="awaiting_confirmation",
                            total_duration_ms=self._ms(started),
                        )
                        yield ChatStreamEvent(kind="interrupt", reply=reply, token=reply.message)
                        return

                    if "tools" in chunk:
                        tool_data = chunk["tools"]
                        for step in tool_data.get("steps", []):
                            accumulated_steps.append(step)
                            yield ChatStreamEvent(kind="step", step=step)

                    if "confirm" in chunk:
                        confirm_data = chunk["confirm"]
                        for step in confirm_data.get("steps", []):
                            accumulated_steps.append(step)
                            yield ChatStreamEvent(kind="step", step=step)

                    if "respond" in chunk:
                        terminal_respond = chunk["respond"]

        except Exception as exc:
            logger.exception("agent stream failed", extra={"thread_id": thread_id})
            from llm.ollama import OllamaUnavailable

            if isinstance(exc, OllamaUnavailable) or isinstance(exc.__cause__, OllamaUnavailable):
                message = (
                    "The language model is currently unavailable. "
                    "Please ensure Ollama is running and try again."
                )
            else:
                message = (
                    "Something went wrong while processing your request. "
                    "Please try rephrasing your question or try again shortly."
                )
            err_reply = ChatReply(
                message=message,
                status="unavailable",
                total_duration_ms=self._ms(started),
            )
            yield ChatStreamEvent(kind="error", reply=err_reply)
            return

        self._awaiting_threads.discard(thread_id)
        status: Literal[
            "answered",
            "awaiting_confirmation",
            "insufficient_evidence",
            "unavailable",
            "iteration_limit",
            "invalid_request",
        ] = "answered"
        if terminal_respond.get("iteration_limit"):
            status = "iteration_limit"
        elif terminal_respond.get("status") == "insufficient_evidence":
            status = "insufficient_evidence"

        final_message = str(terminal_respond.get("response", ""))
        citations = tuple(terminal_respond.get("citations", ()))

        if not streamed_tokens and final_message:
            yield ChatStreamEvent(kind="token", token=final_message)

        reply = ChatReply(
            message=final_message,
            citations=citations,
            steps=tuple(accumulated_steps),
            status=status,
            total_duration_ms=self._ms(started),
        )
        yield ChatStreamEvent(kind="done", reply=reply)

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
        except Exception as exc:
            logger.exception("agent failed", extra={"thread_id": thread_id})
            from llm.ollama import OllamaUnavailable

            if isinstance(exc, OllamaUnavailable) or isinstance(exc.__cause__, OllamaUnavailable):
                message = (
                    "The language model is currently unavailable. "
                    "Please ensure Ollama is running and try again."
                )
            else:
                message = (
                    "Something went wrong while processing your request. "
                    "Please try rephrasing your question or try again shortly."
                )
            return ChatReply(
                message=message,
                status="unavailable",
                total_duration_ms=self._ms(started),
            )
        interrupts = terminal.get("__interrupt__")
        if interrupts:
            self._awaiting_threads.add(thread_id)
            int_obj = interrupts[0]
            val = getattr(int_obj, "value", int_obj)
            summary = val.get("summary", "") if isinstance(val, dict) else str(val)
            return ChatReply(
                message=summary + "? Please answer yes or no.",
                awaiting_confirmation=True,
                preview=summary,
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

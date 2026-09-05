"""Observability module backed by LangSmith for tracing runs, graphs, and spans."""

from __future__ import annotations

import contextvars
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from langsmith import traceable
from langsmith.run_helpers import trace

from domain.models import TimingSpan

logger = logging.getLogger(__name__)

RunType = Literal["tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"]


@dataclass
class SpanRecord:
    name: str
    parent_id: str | None
    started: float
    duration_ms: float = 0.0
    children_ms: float = 0.0


@dataclass
class TraceContext:
    request_id: str
    spans: list[SpanRecord] = field(default_factory=list)
    active_span_stack: list[int] = field(default_factory=list)


current_trace: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "current_trace", default=None
)


def is_tracing_enabled() -> bool:
    """Return True if LangSmith tracing is active and an API key is provided."""
    langsmith_val = os.getenv("LANGSMITH_TRACING")
    langchain_val = os.getenv("LANGCHAIN_TRACING_V2")
    tracing = (
        (langsmith_val is not None and langsmith_val.lower() in ("true", "1", "yes"))
        or (langchain_val is not None and langchain_val.lower() in ("true", "1", "yes"))
    )
    api_key = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
    return bool(tracing and api_key)


def configure_langsmith() -> None:
    """Validate and synchronize LangSmith environment settings.

    Synchronizes LANGSMITH_* and LANGCHAIN_* environment variables.
    If tracing is enabled but no API key is set, disables remote trace submission
    to prevent noisy 401 warnings until configured.
    """
    langsmith_val = os.getenv("LANGSMITH_TRACING")
    langchain_val = os.getenv("LANGCHAIN_TRACING_V2")
    tracing = (
        (langsmith_val is not None and langsmith_val.lower() in ("true", "1", "yes"))
        or (langchain_val is not None and langchain_val.lower() in ("true", "1", "yes"))
    )
    api_key = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
    endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")
    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")

    if endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
        os.environ["LANGSMITH_ENDPOINT"] = endpoint
    if project:
        os.environ["LANGCHAIN_PROJECT"] = project
        os.environ["LANGSMITH_PROJECT"] = project
    if api_key:
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGSMITH_API_KEY"] = api_key

    if tracing and not api_key:
        logger.info(
            "LangSmith tracing enabled but API key is empty. "
            "Disabling remote trace submission until key is provided."
        )
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
    elif tracing and api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"


@contextmanager
def trace_request(
    request_id: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> Iterator[TraceContext]:
    """Scope a customer request under LangSmith tracing with fallback local context."""
    meta: dict[str, Any] = {"request_id": request_id}
    if metadata:
        meta.update(metadata)

    ctx = TraceContext(request_id=request_id)
    token = current_trace.set(ctx)

    try:
        if is_tracing_enabled():
            with trace(name=f"request.{request_id[:8]}", run_type="chain", metadata=meta):
                yield ctx
        else:
            yield ctx
    finally:
        current_trace.reset(token)


@contextmanager
def record_span(
    name: str,
    run_type: RunType = "chain",
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Record a span to LangSmith and update local context timing."""
    ctx = current_trace.get()
    parent_name = None
    span_idx = None
    if ctx is not None:
        parent_name = (
            ctx.spans[ctx.active_span_stack[-1]].name if ctx.active_span_stack else None
        )
        span_idx = len(ctx.spans)
        record = SpanRecord(name=name, parent_id=parent_name, started=time.perf_counter())
        ctx.spans.append(record)
        ctx.active_span_stack.append(span_idx)

    start = time.perf_counter()
    try:
        if is_tracing_enabled():
            with trace(name=name, run_type=run_type, metadata=metadata):
                yield
        else:
            yield
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        if ctx is not None and span_idx is not None:
            ctx.spans[span_idx].duration_ms = duration_ms
            ctx.active_span_stack.pop()
            if ctx.active_span_stack:
                parent_idx = ctx.active_span_stack[-1]
                ctx.spans[parent_idx].children_ms += duration_ms


def get_timing_spans() -> tuple[TimingSpan, ...]:
    """Retrieve fine-grained measured spans for the active trace context."""
    ctx = current_trace.get()
    if ctx is None:
        return ()
    spans: list[TimingSpan] = []
    for s in ctx.spans:
        exclusive_ms = max(0.0, round(s.duration_ms - s.children_ms, 2))
        spans.append(
            TimingSpan(
                name=s.name,
                parent_id=s.parent_id,
                duration_ms=s.duration_ms,
                exclusive_ms=exclusive_ms,
            )
        )
    return tuple(spans)


__all__ = [
    "SpanRecord",
    "TimingSpan",
    "TraceContext",
    "configure_langsmith",
    "current_trace",
    "get_timing_spans",
    "is_tracing_enabled",
    "record_span",
    "trace",
    "trace_request",
    "traceable",
]

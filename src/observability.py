"""Request-scoped execution span collector for fine-grained performance measurement."""

from __future__ import annotations

import contextvars
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from domain.models import TimingSpan


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


@contextmanager
def trace_request(request_id: str) -> Iterator[TraceContext]:
    ctx = TraceContext(request_id=request_id)
    token = current_trace.set(ctx)
    try:
        yield ctx
    finally:
        current_trace.reset(token)


@contextmanager
def record_span(name: str) -> Iterator[None]:
    ctx = current_trace.get()
    if ctx is None:
        yield
        return
    parent_name = (
        ctx.spans[ctx.active_span_stack[-1]].name if ctx.active_span_stack else None
    )
    span_idx = len(ctx.spans)
    record = SpanRecord(name=name, parent_id=parent_name, started=time.perf_counter())
    ctx.spans.append(record)
    ctx.active_span_stack.append(span_idx)
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        record.duration_ms = duration_ms
        ctx.active_span_stack.pop()
        if ctx.active_span_stack:
            parent_idx = ctx.active_span_stack[-1]
            ctx.spans[parent_idx].children_ms += duration_ms


def get_timing_spans() -> tuple[TimingSpan, ...]:
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

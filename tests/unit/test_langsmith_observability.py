from __future__ import annotations

import os
from typing import Any

import pytest
from langgraph.types import Command

from config import Settings
from observability import (
    configure_langsmith,
    get_timing_spans,
    is_tracing_enabled,
    record_span,
    trace_request,
)
from services.chat import AgentService


class CapturingFakeGraph:
    def __init__(self) -> None:
        self.invocations: list[tuple[Any, dict[str, Any] | None]] = []

    def invoke(self, payload: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
        self.invocations.append((payload, config))
        if isinstance(payload, Command):
            return {"response": "Resumed successfully.", "steps": ("cancel_order",)}
        return {"response": "Here is your order.", "steps": ("get_order_status",)}


def test_is_tracing_enabled_checks_flag_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    assert not is_tracing_enabled()

    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "")
    assert not is_tracing_enabled()

    monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_pt_testkey123")
    assert is_tracing_enabled()


def test_configure_langsmith_disables_tracing_if_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

    configure_langsmith()

    assert os.getenv("LANGCHAIN_TRACING_V2") == "false"


def test_configure_langsmith_preserves_tracing_if_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "valid_key")

    configure_langsmith()

    assert os.getenv("LANGCHAIN_TRACING_V2") == "true"


def test_record_span_records_timings_in_trace_request() -> None:
    with trace_request("req-123"):
        with record_span("step.one", run_type="chain"):
            pass
        with record_span("retrieval", run_type="retriever"):
            pass
        spans = get_timing_spans()

    assert len(spans) == 2
    assert spans[0].name == "step.one"
    assert spans[1].name == "retrieval"
    assert spans[0].duration_ms >= 0.0


def test_agent_service_submit_passes_langsmith_config() -> None:
    graph = CapturingFakeGraph()
    service = AgentService(graph)

    reply = service.submit(
        body="Where is my order?", customer_id="CUS-42", thread_id="thread-abc"
    )

    assert reply.status == "answered"
    assert len(graph.invocations) == 1

    payload, config = graph.invocations[0]
    assert payload == {
        "messages": [{"role": "user", "content": "Where is my order?"}],
        "customer_id": "CUS-42",
    }
    assert config is not None
    assert config["run_name"] == "agent.submit"
    assert config["tags"] == ["retail-support"]
    assert config["configurable"] == {"thread_id": "thread-abc"}
    assert config["metadata"]["customer_id"] == "CUS-42"
    assert config["metadata"]["thread_id"] == "thread-abc"
    assert config["metadata"]["request_id"] != ""


def test_agent_service_resume_passes_langsmith_config() -> None:
    graph = CapturingFakeGraph()
    service = AgentService(graph)

    # First submit to establish ownership
    service.submit(body="Cancel ORD-1", customer_id="CUS-42", thread_id="thread-xyz")
    graph.invocations.clear()

    reply = service.resume(thread_id="thread-xyz", customer_id="CUS-42", decision="yes")

    assert reply.status == "answered"
    assert len(graph.invocations) == 1

    payload, config = graph.invocations[0]
    assert isinstance(payload, Command)
    assert payload.resume == "yes"
    assert config is not None
    assert config["run_name"] == "agent.resume"
    assert config["tags"] == ["retail-support"]
    assert config["configurable"] == {"thread_id": "thread-xyz"}
    assert config["metadata"]["customer_id"] == "CUS-42"
    assert config["metadata"]["thread_id"] == "thread-xyz"
    assert config["metadata"]["request_id"] != ""


def test_settings_from_env_loads_langsmith_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "custom-retail")
    monkeypatch.setenv("LANGCHAIN_ENDPOINT", "https://custom.smith.langchain.com")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "secret-test-key")

    settings = Settings.from_env()

    assert settings.langchain_tracing_v2 is True
    assert settings.langchain_project == "custom-retail"
    assert settings.langchain_endpoint == "https://custom.smith.langchain.com"
    assert settings.langchain_api_key == "secret-test-key"

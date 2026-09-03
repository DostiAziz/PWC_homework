from typing import Any

import pytest
from httpx import ReadTimeout
from pydantic import BaseModel

from pwc_support.llm.ollama import (
    OllamaEmbedder,
    OllamaGenerator,
    StructuredOutputInvalid,
)


class Output(BaseModel):
    route: str


class FakeClient:
    def embed(self, *, model: str, input: list[str]) -> dict[str, Any]:
        return {"embeddings": [[1.0, 0.0] for _ in input]}

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        return {"message": {"content": '{"route": "plan"}'}}


def test_generator_wires_context_and_schema_token_limits() -> None:
    class RecordingClient(FakeClient):
        def __init__(self) -> None:
            self.request: dict[str, Any] = {}

        def chat(self, **kwargs: Any) -> dict[str, Any]:
            self.request = kwargs
            return super().chat(**kwargs)

    client = RecordingClient()
    OllamaGenerator(
        client,
        "risk-model:1",
        num_ctx=4096,
        schema_tokens=128,
    ).structured(system="Return JSON.", user="Classify this.", schema=Output)

    assert client.request["options"] == {
        "temperature": 0.0,
        "num_predict": 128,
        "num_ctx": 4096,
    }


def test_ollama_adapters_use_explicit_models() -> None:
    client = FakeClient()

    embeddings = OllamaEmbedder(client, "nomic-embed-text").embed(["hello"])
    answer = OllamaGenerator(client, "gpt-oss:20b").structured(
        system="Return JSON.", user="Classify this.", schema=Output
    )

    assert embeddings == [[1.0, 0.0]]
    assert answer.route == "plan"


def test_native_client_binds_the_transport_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def client_factory(*, host: str, timeout: float) -> FakeClient:
        captured.update(host=host, timeout=timeout)
        return FakeClient()

    monkeypatch.setattr("pwc_support.llm.ollama.ollama.Client", client_factory)

    generator = OllamaGenerator.from_connection(
        host="http://127.0.0.1:11434",
        model="risk-model:1",
        request_timeout_seconds=2.5,
    )

    assert captured == {"host": "http://127.0.0.1:11434", "timeout": 2.5}
    assert generator.request_timeout_seconds == 2.5
    assert generator.model == "risk-model:1"


def test_invalid_structured_response_has_a_dedicated_boundary_error() -> None:
    class InvalidClient(FakeClient):
        def chat(self, **kwargs: Any) -> dict[str, Any]:
            return {"message": {"content": "not json"}}

    with pytest.raises(StructuredOutputInvalid):
        OllamaGenerator(InvalidClient(), "risk-model:1").structured(
            system="Return JSON.",
            user="Classify this.",
            schema=Output,
        )


def test_native_transport_timeout_is_normalized_without_a_worker_thread() -> None:
    class TimeoutClient(FakeClient):
        def chat(self, **kwargs: Any) -> dict[str, Any]:
            raise ReadTimeout("read timed out")

    generator = OllamaGenerator(
        TimeoutClient(),
        "risk-model:1",
        request_timeout_seconds=2.5,
    )

    with pytest.raises(TimeoutError):
        generator.structured(system="Return JSON.", user="Classify this.", schema=Output)

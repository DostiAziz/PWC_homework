from typing import Any

import pytest
from httpx import ReadTimeout
from pydantic import BaseModel

from pwc_support.llm.ollama import (
    OllamaGateway,
    OllamaUnavailable,
    StructuredOutputInvalid,
)


class Output(BaseModel):
    route: str


class RecordingClient:
    def __init__(self) -> None:
        self.chat_request: dict[str, Any] = {}

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.chat_request = kwargs
        return {"message": {"content": '{"route":"catalogue"}'}}

    def embed(self, **_: Any) -> dict[str, Any]:
        return {"embeddings": [[1, 0], [0, 1]]}


def test_structured_generation_sends_schema_and_validates_response() -> None:
    client = RecordingClient()
    gateway = OllamaGateway(
        client,
        generation_model="gpt-oss:20b",
        embedding_model="nomic-embed-text",
        schema_tokens=128,
        num_ctx=4096,
    )

    output = gateway.structured(system="Classify", user="offers", schema=Output)

    assert output.route == "catalogue"
    assert client.chat_request["format"] == Output.model_json_schema()
    assert client.chat_request["options"] == {
        "temperature": 0.0,
        "num_predict": 128,
        "num_ctx": 4096,
    }


def test_text_and_embeddings_use_explicit_models() -> None:
    client = RecordingClient()
    gateway = OllamaGateway(
        client,
        generation_model="gpt-oss:20b",
        embedding_model="nomic-embed-text",
    )

    assert gateway.text(system="Answer", user="question") == '{"route":"catalogue"}'
    assert gateway.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_transport_timeout_has_one_boundary_error() -> None:
    class TimeoutClient(RecordingClient):
        def chat(self, **kwargs: Any) -> dict[str, Any]:
            raise ReadTimeout("timed out")

    gateway = OllamaGateway(
        TimeoutClient(),
        generation_model="gpt-oss:20b",
        embedding_model="nomic-embed-text",
    )

    with pytest.raises(OllamaUnavailable, match="Ollama request failed"):
        gateway.text(system="Answer", user="question")


def test_invalid_schema_output_has_one_boundary_error() -> None:
    class InvalidClient(RecordingClient):
        def chat(self, **kwargs: Any) -> dict[str, Any]:
            return {"message": {"content": "not JSON"}}

    gateway = OllamaGateway(
        InvalidClient(),
        generation_model="gpt-oss:20b",
        embedding_model="nomic-embed-text",
    )

    with pytest.raises(StructuredOutputInvalid):
        gateway.structured(system="Classify", user="offers", schema=Output)


def test_chat_with_tools_normalizes_tool_calls() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.captured: dict[str, Any] = {}

        def chat(self, **kwargs: Any) -> dict[str, Any]:
            self.captured = kwargs
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_order_status",
                                "arguments": {"order_id": "ORD-1"},
                            }
                        }
                    ],
                }
            }

    client = FakeClient()
    gw = OllamaGateway(client, generation_model="m", embedding_model="e")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_order_status",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                },
            },
        }
    ]
    msg = gw.chat_with_tools(
        messages=[{"role": "user", "content": "where is ORD-1"}], tools=tools
    )

    assert msg["tool_calls"] == [{"name": "get_order_status", "arguments": {"order_id": "ORD-1"}}]
    assert client.captured["tools"] == tools
    assert client.captured["messages"][0]["content"] == "where is ORD-1"


def test_chat_with_tools_returns_empty_tool_calls_for_plain_answer() -> None:
    class FakeClient:
        def chat(self, **kwargs: Any) -> dict[str, Any]:
            return {"message": {"role": "assistant", "content": "Hello."}}

    gw = OllamaGateway(FakeClient(), generation_model="m", embedding_model="e")
    msg = gw.chat_with_tools(messages=[{"role": "user", "content": "hi"}], tools=[])
    assert msg["content"] == "Hello."
    assert msg["tool_calls"] == []


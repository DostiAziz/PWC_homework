from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from retail_support.llm.embeddings import HuggingFaceEmbedder
from retail_support.llm.ollama import (
    ChatOpenAIAdapter,
    OllamaGateway,
    OllamaUnavailable,
    StructuredOutputInvalid,
)


class Output(BaseModel):
    route: str


class FakeStructuredModel:
    def __init__(self, output: Any) -> None:
        self.output = output

    def invoke(self, messages: list[BaseMessage]) -> Any:
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


class FakeChatModel:
    def __init__(self, response: Any = None) -> None:
        self.response = response or AIMessage(content="Hello from fake")
        self.bound_tools: list[dict[str, Any]] | None = None
        self.captured_messages: list[BaseMessage] = []

    def invoke(self, messages: list[BaseMessage]) -> Any:
        if isinstance(self.response, Exception):
            raise self.response
        self.captured_messages = list(messages)
        return self.response

    def bind_tools(self, tools: list[dict[str, Any]]) -> Any:
        self.bound_tools = tools
        return self

    def with_structured_output(self, schema: type[Any]) -> Any:
        return FakeStructuredModel(Output(route="catalogue"))


class FakeEmbeddingsBackend:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[1.0, 0.0] for _ in texts]


def test_structured_generation_returns_validated_model() -> None:
    fake_llm = FakeChatModel()
    gateway = ChatOpenAIAdapter(llm=fake_llm)

    output = gateway.structured(system="Classify", user="offers", schema=Output)

    assert isinstance(output, Output)
    assert output.route == "catalogue"


def test_text_generation_returns_string() -> None:
    fake_llm = FakeChatModel(response=AIMessage(content="Order shipped."))
    gateway = ChatOpenAIAdapter(llm=fake_llm)

    assert gateway.text(system="System", user="Where is ORD-1?") == "Order shipped."
    assert len(fake_llm.captured_messages) == 2


def test_transport_error_raises_ollama_unavailable() -> None:
    fake_llm = FakeChatModel(response=RuntimeError("connection refused"))
    gateway = ChatOpenAIAdapter(llm=fake_llm)

    with pytest.raises(OllamaUnavailable, match="LLM request failed"):
        gateway.text(system="System", user="Hello")


def test_invalid_structured_output_raises_structured_output_invalid() -> None:
    class FailingFake(FakeChatModel):
        def with_structured_output(self, schema: type[Any]) -> Any:
            return FakeStructuredModel({"invalid_field": "unknown"})

    gateway = ChatOpenAIAdapter(llm=FailingFake())

    with pytest.raises(StructuredOutputInvalid):
        gateway.structured(system="Classify", user="offers", schema=Output)


def test_chat_with_tools_returns_ai_message() -> None:
    response = AIMessage(
        content="",
        tool_calls=[{"name": "get_order_status", "args": {"order_id": "ORD-1"}, "id": "c1"}],
    )
    fake_llm = FakeChatModel(response=response)
    gw = OllamaGateway(llm=fake_llm)
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

    msg = gw.chat_with_tools(messages=[HumanMessage(content="where is ORD-1")], tools=tools)

    assert msg.tool_calls[0]["name"] == "get_order_status"
    assert msg.tool_calls[0]["args"] == {"order_id": "ORD-1"}
    assert msg.tool_calls[0]["id"] == "c1"
    assert fake_llm.bound_tools == tools


def test_hugging_face_embedder_wraps_backend() -> None:
    fake_backend = FakeEmbeddingsBackend()
    embedder = HuggingFaceEmbedder(embeddings_instance=fake_backend)

    results = embedder.embed(["test doc 1", "test doc 2"])

    assert len(results) == 2
    assert results[0] == [1.0, 0.0]
    assert fake_backend.calls == [["test doc 1", "test doc 2"]]


def test_chat_openai_adapter_embed_delegates_to_embedder() -> None:
    fake_backend = FakeEmbeddingsBackend()
    embedder = HuggingFaceEmbedder(embeddings_instance=fake_backend)
    gw = ChatOpenAIAdapter(llm=FakeChatModel(), embedder=embedder)

    results = gw.embed(["sample"])

    assert results == [[1.0, 0.0]]

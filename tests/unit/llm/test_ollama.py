from threading import Event, Lock, Thread
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from llm.embeddings import (
    HuggingFaceEmbedder,
    embedding_dimension,
    validate_collection_dimension,
)
from llm.ollama import (
    ChatOpenAIAdapter,
    OllamaGateway,
    OllamaUnavailable,
    StructuredOutputInvalid,
    get_chat_model,
)


class Output(BaseModel):
    route: str


class FakeStructuredModel:
    def __init__(self, output: Any) -> None:
        self.output = output

    def invoke(self, messages: list[BaseMessage], config: Any = None, **kwargs: Any) -> Any:
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


def test_factory_applies_output_budget_and_finite_retries() -> None:
    model = get_chat_model(max_output_tokens=73, max_parallel_generations=1)

    assert model.backend._get_invocation_params()["max_completion_tokens"] == 73
    assert model.backend.max_retries == 2


class EventControlledModel(FakeChatModel):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail
        self.entered = Event()
        self.release = Event()
        self.lock = Lock()
        self.active = 0
        self.peak_active = 0
        self.configs: list[Any] = []

    def invoke(self, messages: list[BaseMessage], config: Any = None, **kwargs: Any) -> Any:
        with self.lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            self.configs.append(config)
        self.entered.set()
        self.release.wait(timeout=2)
        try:
            if self.fail:
                raise RuntimeError("backend failed")
            return self.response
        finally:
            with self.lock:
                self.active -= 1


def test_bound_and_unbound_calls_share_generation_limit_and_forward_config() -> None:
    backend = EventControlledModel()
    model = get_chat_model(
        max_output_tokens=64,
        max_parallel_generations=1,
        request_timeout_seconds=1,
        backend=backend,
    )
    bound = model.bind_tools([])
    first = Thread(target=model.invoke, args=([HumanMessage(content="one")], {"tags": ["one"]}))
    second = Thread(target=bound.invoke, args=([HumanMessage(content="two")], {"tags": ["two"]}))

    first.start()
    assert backend.entered.wait(timeout=1)
    second.start()
    assert backend.peak_active == 1
    backend.release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert backend.peak_active == 1
    assert backend.configs == [{"tags": ["one"]}, {"tags": ["two"]}]


def test_generation_exception_releases_shared_slot() -> None:
    failing = EventControlledModel(fail=True)
    failing.release.set()
    model = get_chat_model(
        max_output_tokens=64,
        max_parallel_generations=1,
        request_timeout_seconds=1,
        backend=failing,
    )

    with pytest.raises(RuntimeError, match="backend failed"):
        model.invoke([HumanMessage(content="first")])

    succeeding = EventControlledModel()
    succeeding.release.set()
    model.backend = succeeding
    assert model.invoke([HumanMessage(content="second")]).content == "Hello from fake"


def test_embedding_diagnostic_rejects_incompatible_collection() -> None:
    class CollectionModel:
        dimension = 768

    class Collection:
        name = "retail_support"

        def get_model(self) -> CollectionModel:
            return CollectionModel()

    with pytest.raises(ValueError, match=r"expected 768 dimensions.*produced 2.*rebuild"):
        validate_collection_dimension(
            Collection(),
            embedding_dimension(HuggingFaceEmbedder(embeddings_instance=FakeEmbeddingsBackend())),
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

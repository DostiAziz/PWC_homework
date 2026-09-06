from threading import Event, Lock, Thread
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from llm.embeddings import (
    embedding_dimension,
    validate_collection_dimension,
)
from llm.ollama import (
    LimitedChatModel,
    OllamaUnavailable,
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

    def invoke(self, messages: list[BaseMessage], config: Any = None, **kwargs: Any) -> Any:
        if isinstance(self.response, Exception):
            raise self.response
        self.captured_messages = list(messages)
        return self.response

    def stream(self, messages: list[BaseMessage], config: Any = None, **kwargs: Any) -> Any:
        if isinstance(self.response, Exception):
            raise self.response
        self.captured_messages = list(messages)
        yield self.response

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

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_limited_chat_model_invoke_returns_ai_message() -> None:
    fake_llm = FakeChatModel(response=AIMessage(content="Order shipped."))
    model = LimitedChatModel(fake_llm)

    response = model.invoke([HumanMessage(content="Where is ORD-1?")])

    assert response.content == "Order shipped."
    assert len(fake_llm.captured_messages) == 1


def test_limited_chat_model_bind_tools_propagates() -> None:
    fake_llm = FakeChatModel()
    model = LimitedChatModel(fake_llm)
    tools = [{"type": "function", "function": {"name": "test_tool"}}]

    bound = model.bind_tools(tools)
    assert isinstance(bound, LimitedChatModel)
    assert fake_llm.bound_tools == tools


def test_limited_chat_model_times_out_when_concurrency_saturated() -> None:
    backend = EventControlledModel()
    model = LimitedChatModel(backend, max_parallel_generations=1, acquisition_timeout_seconds=0.01)

    first = Thread(target=model.invoke, args=([HumanMessage(content="one")],))
    first.start()
    assert backend.entered.wait(timeout=1)

    with pytest.raises(OllamaUnavailable, match="Timed out waiting"):
        model.invoke([HumanMessage(content="two")])

    backend.release.set()
    first.join(timeout=2)


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


def test_limited_chat_model_stream_yields_chunks_and_releases_limiter() -> None:
    fake = FakeChatModel(response=AIMessage(content="Chunk response"))
    model = get_chat_model(
        max_output_tokens=64,
        max_parallel_generations=1,
        request_timeout_seconds=1,
        backend=fake,
    )
    chunks = list(model.stream([HumanMessage(content="hi")]))
    assert len(chunks) == 1
    assert chunks[0].content == "Chunk response"
    # Verify limiter was released and another call succeeds
    assert model.invoke([HumanMessage(content="hi again")]).content == "Chunk response"


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
            embedding_dimension(FakeEmbeddingsBackend()),
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

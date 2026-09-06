from __future__ import annotations

from collections.abc import Sequence
from threading import BoundedSemaphore
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from observability import record_span


class OllamaUnavailable(RuntimeError):
    pass


def get_chat_model(
    model_name: str = "gpt-oss:20b",
    *,
    base_url: str = "http://127.0.0.1:11434",
    temperature: float = 0.0,
    request_timeout_seconds: float = 120.0,
    max_output_tokens: int = 512,
    max_parallel_generations: int = 1,
    backend: Any | None = None,
    **kwargs: Any,
) -> LimitedChatModel:
    """Create a ChatOpenAI model pointed at Ollama's /v1 endpoint."""
    v1_url = base_url.rstrip("/")
    if not v1_url.endswith("/v1"):
        v1_url = f"{v1_url}/v1"
    streaming = kwargs.pop("streaming", True)
    resolved_backend = backend or ChatOpenAI(
        model=model_name,
        base_url=v1_url,
        api_key=SecretStr("ollama"),
        temperature=temperature,
        timeout=request_timeout_seconds,
        max_completion_tokens=max_output_tokens,
        max_retries=2,
        streaming=streaming,
        **kwargs,
    )
    return LimitedChatModel(
        resolved_backend,
        max_parallel_generations=max_parallel_generations,
        acquisition_timeout_seconds=request_timeout_seconds,
    )


class LimitedChatModel:
    """Serialize local generation through a limiter shared by all bound variants."""

    def __init__(
        self,
        backend: Any,
        *,
        max_parallel_generations: int = 1,
        acquisition_timeout_seconds: float = 120.0,
        limiter: BoundedSemaphore | None = None,
    ) -> None:
        self.backend = backend
        self._max_parallel_generations = max_parallel_generations
        self._acquisition_timeout_seconds = acquisition_timeout_seconds
        self._limiter = limiter or BoundedSemaphore(max_parallel_generations)

    def invoke(
        self,
        messages: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        with record_span("generation.queue", run_type="chain"):
            acquired = self._limiter.acquire(timeout=self._acquisition_timeout_seconds)
            if not acquired:
                raise OllamaUnavailable(
                    "Timed out waiting for an available local generation slot"
                )
        try:
            with record_span("generation.invoke", run_type="llm"):
                if config is not None:
                    return self.backend.invoke(messages, config=config, **kwargs)
                return self.backend.invoke(messages, **kwargs)
        finally:
            self._limiter.release()

    def stream(
        self,
        messages: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        with record_span("generation.queue", run_type="chain"):
            acquired = self._limiter.acquire(timeout=self._acquisition_timeout_seconds)
            if not acquired:
                raise OllamaUnavailable(
                    "Timed out waiting for an available local generation slot"
                )
        try:
            with record_span("generation.stream", run_type="llm"):
                if config is not None:
                    yield from self.backend.stream(messages, config=config, **kwargs)
                else:
                    yield from self.backend.stream(messages, **kwargs)
        finally:
            self._limiter.release()

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> LimitedChatModel:
        return LimitedChatModel(
            self.backend.bind_tools(tools, **kwargs),
            max_parallel_generations=self._max_parallel_generations,
            acquisition_timeout_seconds=self._acquisition_timeout_seconds,
            limiter=self._limiter,
        )

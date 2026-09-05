from __future__ import annotations

from collections.abc import Sequence
from threading import BoundedSemaphore
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr, ValidationError

from observability import record_span

ModelT = TypeVar("ModelT", bound=BaseModel)


class OllamaUnavailable(RuntimeError):
    pass


class StructuredOutputInvalid(ValueError):
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
    resolved_backend = backend or ChatOpenAI(
        model=model_name,
        base_url=v1_url,
        api_key=SecretStr("ollama"),
        temperature=temperature,
        timeout=request_timeout_seconds,
        max_completion_tokens=max_output_tokens,
        max_retries=2,
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
        max_parallel_generations: int,
        acquisition_timeout_seconds: float,
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
        with record_span("generation.queue"):
            acquired = self._limiter.acquire(timeout=self._acquisition_timeout_seconds)
            if not acquired:
                raise OllamaUnavailable(
                    "Timed out waiting for an available local generation slot"
                )
        try:
            with record_span("generation.invoke"):
                return self.backend.invoke(messages, config=config, **kwargs)
        finally:
            self._limiter.release()

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> LimitedChatModel:
        return LimitedChatModel(
            self.backend.bind_tools(tools, **kwargs),
            max_parallel_generations=self._max_parallel_generations,
            acquisition_timeout_seconds=self._acquisition_timeout_seconds,
            limiter=self._limiter,
        )


class ChatOpenAIAdapter:
    """OpenAI-compatible chat adapter pointing to local Ollama (/v1)."""

    def __init__(
        self,
        *,
        generation_model: str = "gpt-oss:20b",
        base_url: str = "http://127.0.0.1:11434",
        temperature: float = 0.0,
        request_timeout_seconds: float = 120.0,
        llm: Any | None = None,
        embedder: Any | None = None,
        **_: Any,
    ) -> None:
        self.generation_model = generation_model
        self.embedder = embedder
        if llm is not None:
            self.llm = llm
        else:
            self.llm = get_chat_model(
                model_name=generation_model,
                base_url=base_url,
                temperature=temperature,
                request_timeout_seconds=request_timeout_seconds,
            )

    def bind_tools(self, tools: list[dict[str, Any]]) -> Any:
        return self.llm.bind_tools(tools)

    def invoke(self, messages: list[BaseMessage] | list[Any]) -> Any:
        try:
            return self.llm.invoke(messages)
        except Exception as error:
            raise OllamaUnavailable("LLM request failed") from error

    def chat_with_tools(
        self,
        *,
        messages: list[BaseMessage] | list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Any:
        try:
            bound = self.llm.bind_tools(tools) if tools else self.llm
            return bound.invoke(messages)
        except Exception as error:
            raise OllamaUnavailable("LLM tool chat failed") from error

    def text(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        try:
            res = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            content = res.content if isinstance(res.content, str) else str(res.content)
            return content.strip()
        except Exception as error:
            raise OllamaUnavailable("LLM request failed") from error

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[ModelT],
        temperature: float = 0.0,
    ) -> ModelT:
        try:
            structured_llm = self.llm.with_structured_output(schema)
            output = structured_llm.invoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
            if isinstance(output, schema):
                return output
            if isinstance(output, dict):
                return schema.model_validate(output)
            raise StructuredOutputInvalid("Output does not match expected schema")
        except ValidationError as error:
            raise StructuredOutputInvalid("Response does not match schema") from error
        except Exception as error:
            if isinstance(error, StructuredOutputInvalid):
                raise
            raise OllamaUnavailable("Structured output request failed") from error

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.embedder is not None:
            vectors = self.embedder.embed(texts)
            return [[float(v) for v in vector] for vector in vectors]
        raise OllamaUnavailable("No embedder configured")


# Backward compatibility aliases
OllamaGateway = ChatOpenAIAdapter

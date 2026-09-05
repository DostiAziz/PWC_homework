from __future__ import annotations

from threading import BoundedSemaphore
from typing import Any, TypeVar

import ollama
from httpx import HTTPError
from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


class OllamaUnavailable(RuntimeError):
    pass


class StructuredOutputInvalid(ValueError):
    pass


class OllamaGateway:
    def __init__(
        self,
        client: Any,
        *,
        generation_model: str,
        embedding_model: str,
        num_ctx: int | None = None,
        schema_tokens: int = 256,
        max_parallel_generations: int = 1,
    ) -> None:
        self.client = client
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.num_ctx = num_ctx
        self.schema_tokens = schema_tokens
        self.generation_slots = BoundedSemaphore(max_parallel_generations)

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self.client.embed(model=self.embedding_model, input=texts)
            return [[float(value) for value in vector] for vector in response["embeddings"]]
        except (HTTPError, ollama.RequestError, ollama.ResponseError, KeyError, TypeError) as error:
            raise OllamaUnavailable("Ollama embedding request failed") from error

    def text(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        return self._chat(
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=None,
        )

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[ModelT],
        temperature: float = 0.0,
    ) -> ModelT:
        content = self._chat(
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=self.schema_tokens,
            response_format=schema.model_json_schema(),
        )
        try:
            return schema.model_validate_json(content)
        except ValidationError as error:
            raise StructuredOutputInvalid("Ollama response does not match schema") from error

    def chat_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        options: dict[str, float | int] = {"temperature": temperature, "num_predict": max_tokens}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        request: dict[str, Any] = {
            "model": self.generation_model,
            "messages": messages,
            "options": options,
        }
        if tools:
            request["tools"] = tools
        try:
            with self.generation_slots:
                response = self.client.chat(**request)
            message = response["message"]
        except (HTTPError, ollama.RequestError, ollama.ResponseError, KeyError, TypeError) as error:
            raise OllamaUnavailable("Ollama tool chat failed") from error
        raw_calls = message.get("tool_calls") or []
        tool_calls = [
            {
                "name": call["function"]["name"],
                "arguments": dict(call["function"].get("arguments") or {}),
            }
            for call in raw_calls
        ]
        return {
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": tool_calls,
        }

    def _chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
    ) -> str:
        options: dict[str, float | int] = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        request: dict[str, Any] = {
            "model": self.generation_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": options,
        }
        if response_format is not None:
            request["format"] = response_format
        try:
            with self.generation_slots:
                response = self.client.chat(**request)
            content = response["message"]["content"]
        except (HTTPError, ollama.RequestError, ollama.ResponseError, KeyError, TypeError) as error:
            raise OllamaUnavailable("Ollama request failed") from error
        if not isinstance(content, str):
            raise OllamaUnavailable("Ollama response has no text content")
        return content.strip()

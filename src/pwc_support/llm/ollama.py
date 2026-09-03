from __future__ import annotations

import json
from threading import BoundedSemaphore
from typing import Any, TypeVar

import ollama
from httpx import TimeoutException
from pydantic import BaseModel, ValidationError

from pwc_support.rag.ingest import CorpusDocument

ModelT = TypeVar("ModelT", bound=BaseModel)


class StructuredOutputInvalid(ValueError):
    """A model response could not be decoded and validated against its schema."""


class OllamaEmbedder:
    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embed(model=self.model, input=texts)
        return [[float(value) for value in vector] for vector in response["embeddings"]]


class OllamaGenerator:
    def __init__(
        self,
        client: Any,
        model: str,
        *,
        temperature: float = 0.0,
        request_timeout_seconds: float | None = None,
        num_ctx: int | None = None,
        schema_tokens: int = 256,
        max_parallel_generations: int = 1,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.request_timeout_seconds = request_timeout_seconds
        self.num_ctx = num_ctx
        self.schema_tokens = schema_tokens
        self._generation_slots = BoundedSemaphore(max_parallel_generations)

    @classmethod
    def from_connection(
        cls,
        *,
        host: str,
        model: str,
        request_timeout_seconds: float,
        temperature: float = 0.0,
        num_ctx: int | None = None,
        schema_tokens: int = 256,
        max_parallel_generations: int = 1,
    ) -> OllamaGenerator:
        """Build an adapter whose HTTP transport enforces the request deadline."""
        client = ollama.Client(host=host, timeout=request_timeout_seconds)
        return cls(
            client,
            model,
            temperature=temperature,
            request_timeout_seconds=request_timeout_seconds,
            num_ctx=num_ctx,
            schema_tokens=schema_tokens,
            max_parallel_generations=max_parallel_generations,
        )

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[ModelT],
        temperature: float | None = None,
    ) -> ModelT:
        selected_temperature = self.temperature if temperature is None else temperature
        try:
            with self._generation_slots:
                response = self.client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    format=schema.model_json_schema(),
                    options=self._options(
                        temperature=selected_temperature, max_tokens=self.schema_tokens
                    ),
                )
        except TimeoutException as error:
            raise TimeoutError("structured Ollama request timed out") from error

        try:
            content = response["message"]["content"]
        except (KeyError, TypeError) as error:
            raise StructuredOutputInvalid("structured response has no content") from error
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError) as error:
            raise StructuredOutputInvalid("structured response is not valid JSON") from error
        try:
            return schema.model_validate(payload)
        except ValidationError as error:
            raise StructuredOutputInvalid("structured response does not match schema") from error

    def text(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        try:
            with self._generation_slots:
                response = self.client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    options=self._options(temperature=temperature, max_tokens=max_tokens),
                )
        except TimeoutException as error:
            raise TimeoutError("text Ollama request timed out") from error
        return str(response["message"]["content"]).strip()

    def _options(self, *, temperature: float, max_tokens: int) -> dict[str, float | int]:
        options: dict[str, float | int] = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        return options


class OllamaContextualizer:
    """Generate Anthropic-style chunk context locally during ingestion."""

    def __init__(self, generator: OllamaGenerator, *, max_document_chars: int = 24000) -> None:
        self.generator = generator
        self.max_document_chars = max_document_chars

    def contextualize(self, *, document: CorpusDocument, heading: str, chunk: str) -> str:
        bounded_document = document.text[: self.max_document_chars]
        return self.generator.text(
            system=(
                "Write only a short context that situates the chunk in its source document "
                "for retrieval. Preserve names, sector, service, territory and time scope."
            ),
            user=(
                f"<document title='{document.title}'>\n{bounded_document}\n</document>\n"
                f"<heading>{heading}</heading>\n<chunk>{chunk}</chunk>"
            ),
            max_tokens=100,
            temperature=0.0,
        )

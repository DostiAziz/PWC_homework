from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, TypeVar

from pydantic import BaseModel

from pwc_support.rag.ingest import CorpusDocument

ModelT = TypeVar("ModelT", bound=BaseModel)


class OllamaEmbedder:
    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embed(model=self.model, input=texts)
        return [[float(value) for value in vector] for vector in response["embeddings"]]


class OllamaGenerator:
    def __init__(self, client: Any, model: str, *, temperature: float = 0.0) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[ModelT],
        temperature: float | None = None,
        timeout_seconds: float | None = None,
    ) -> ModelT:
        selected_temperature = self.temperature if temperature is None else temperature

        def request() -> Any:
            return self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                format=schema.model_json_schema(),
                options={"temperature": selected_temperature},
            )

        if timeout_seconds is None:
            response = request()
        else:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ollama-structured")
            future = executor.submit(request)
            try:
                response = future.result(timeout=timeout_seconds)
            except FutureTimeoutError as error:
                future.cancel()
                raise TimeoutError("structured Ollama request timed out") from error
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        content = response["message"]["content"]
        return schema.model_validate(json.loads(content))

    def text(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return str(response["message"]["content"]).strip()


class OllamaContextualizer:
    """Generate Anthropic-style chunk context locally during ingestion."""

    def __init__(self, generator: OllamaGenerator, *, max_document_chars: int = 24000) -> None:
        self.generator = generator
        self.max_document_chars = max_document_chars

    def contextualize(
        self, *, document: CorpusDocument, heading: str, chunk: str
    ) -> str:
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

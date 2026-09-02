from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel

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

    def structured(self, *, system: str, user: str, schema: type[ModelT]) -> ModelT:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            format=schema.model_json_schema(),
            options={"temperature": self.temperature},
        )
        content = response["message"]["content"]
        return schema.model_validate(json.loads(content))

    def text(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            options={"temperature": 0.2, "num_predict": max_tokens},
        )
        return str(response["message"]["content"]).strip()

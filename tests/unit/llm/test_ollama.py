from typing import Any

from pydantic import BaseModel

from pwc_support.llm.ollama import OllamaEmbedder, OllamaGenerator


class Output(BaseModel):
    route: str


class FakeClient:
    def embed(self, *, model: str, input: list[str]) -> dict[str, Any]:
        return {"embeddings": [[1.0, 0.0] for _ in input]}

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        return {"message": {"content": '{"route": "plan"}'}}


def test_ollama_adapters_use_explicit_models() -> None:
    client = FakeClient()

    embeddings = OllamaEmbedder(client, "nomic-embed-text").embed(["hello"])
    answer = OllamaGenerator(client, "gpt-oss:20b").structured(
        system="Return JSON.", user="Classify this.", schema=Output
    )

    assert embeddings == [[1.0, 0.0]]
    assert answer.route == "plan"

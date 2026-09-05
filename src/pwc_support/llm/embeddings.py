from __future__ import annotations

from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings


class HuggingFaceEmbedder:
    """Adapter for Hugging Face embeddings fulfilling the Embedder protocol."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        model_kwargs: dict[str, Any] | None = None,
        encode_kwargs: dict[str, Any] | None = None,
        embeddings_instance: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.embeddings = (
            embeddings_instance
            if embeddings_instance is not None
            else HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs or {},
                encode_kwargs=encode_kwargs or {"normalize_embeddings": True},
            )
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.embeddings.embed_documents(texts)
        return [[float(v) for v in vector] for vector in embeddings]

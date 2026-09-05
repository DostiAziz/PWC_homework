from __future__ import annotations

from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    *,
    model_kwargs: dict[str, Any] | None = None,
    encode_kwargs: dict[str, Any] | None = None,
) -> HuggingFaceEmbeddings:
    """Instantiate HuggingFaceEmbeddings directly with normalized cosine embeddings."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs or {},
        encode_kwargs=encode_kwargs or {"normalize_embeddings": True},
    )


class HuggingFaceEmbedder(HuggingFaceEmbeddings):
    """Backward-compatible HuggingFaceEmbeddings fulfilling both LangChain and legacy protocols."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        model_kwargs: dict[str, Any] | None = None,
        encode_kwargs: dict[str, Any] | None = None,
        embeddings_instance: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if embeddings_instance is not None:
            self._backend: Any = embeddings_instance
        else:
            super().__init__(
                model_name=model_name,
                model_kwargs=model_kwargs or {},
                encode_kwargs=encode_kwargs or {"normalize_embeddings": True},
                **kwargs,
            )
            self._backend = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._backend is not None:
            return [[float(v) for v in vec] for vec in self._backend.embed_documents(texts)]
        return [[float(v) for v in vec] for vec in super().embed_documents(texts)]

    def embed_query(self, text: str) -> list[float]:
        if self._backend is not None:
            if hasattr(self._backend, "embed_query"):
                return [float(v) for v in self._backend.embed_query(text)]
            return [float(v) for v in self._backend.embed_documents([text])[0]]
        return [float(v) for v in super().embed_query(text)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


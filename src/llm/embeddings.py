from __future__ import annotations

from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings


def embedding_dimension(embedder: Any) -> int:
    """Load the configured backend and return the dimension of a diagnostic vector."""
    vector = embedder.embed_query("embedding dimension diagnostic")
    if not vector:
        raise ValueError("Configured embedding model returned an empty diagnostic vector")
    return len(vector)


def validate_collection_dimension(
    collection: Any,
    dimension: int,
    *,
    model_name: str,
) -> None:
    """Reject reuse of an initialized Chroma collection with incompatible vectors."""
    collection_model = collection.get_model()
    stored_dimension = getattr(collection_model, "dimension", None)
    if stored_dimension is None or int(stored_dimension) == dimension:
        return
    name = getattr(collection, "name", "configured collection")
    raise ValueError(
        f"Chroma collection {name!r} expected {stored_dimension} dimensions, but embedding "
        f"model {model_name!r} produced {dimension}. Delete and rebuild the collection with "
        "`python scripts/ingest_corpus.py`."
    )


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

"""Local LLM and embedding integrations."""

from llm.embeddings import HuggingFaceEmbedder, get_embeddings
from llm.ollama import (
    ChatOpenAIAdapter,
    OllamaGateway,
    OllamaUnavailable,
    StructuredOutputInvalid,
    get_chat_model,
)

__all__ = [
    "ChatOpenAIAdapter",
    "HuggingFaceEmbedder",
    "OllamaGateway",
    "OllamaUnavailable",
    "StructuredOutputInvalid",
    "get_chat_model",
    "get_embeddings",
]

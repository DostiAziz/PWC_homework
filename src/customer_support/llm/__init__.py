"""Local LLM and embedding integrations."""

from customer_support.llm.embeddings import HuggingFaceEmbedder, get_embeddings
from customer_support.llm.ollama import (
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

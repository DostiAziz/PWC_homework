"""Local LLM and embedding integrations."""

from pwc_support.llm.embeddings import HuggingFaceEmbedder
from pwc_support.llm.ollama import (
    ChatOllamaAdapter,
    ChatOpenAIAdapter,
    OllamaGateway,
    OllamaUnavailable,
    StructuredOutputInvalid,
)

__all__ = [
    "ChatOllamaAdapter",
    "ChatOpenAIAdapter",
    "HuggingFaceEmbedder",
    "OllamaGateway",
    "OllamaUnavailable",
    "StructuredOutputInvalid",
]

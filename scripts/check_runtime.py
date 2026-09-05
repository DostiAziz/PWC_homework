from __future__ import annotations

import ollama

from config import Settings
from llm.embeddings import embedding_dimension, get_embeddings, validate_collection_dimension
from rag import ChromaKnowledgeBase, chroma_client

settings = Settings.from_env()
client = ollama.Client(host=settings.ollama_base_url)
models = {str(item.model) for item in client.list().models if item.model}
normalized_models = {name.removesuffix(":latest") for name in models}
required = {settings.generation_model}
normalized_required = {name.removesuffix(":latest") for name in required}
missing = sorted(normalized_required - normalized_models)
print(f"Ollama endpoint: {settings.ollama_base_url}")
print(f"Required Ollama generation model: {', '.join(sorted(required))}")
print(f"HuggingFace embedding model: {settings.embedding_model}")
if missing:
    raise SystemExit(f"Missing local Ollama models: {', '.join(missing)}")
embedder = get_embeddings(model_name=settings.embedding_model)
dimension = embedding_dimension(embedder)
store = ChromaKnowledgeBase(chroma_client(settings), settings.collection_name, embedder)
validate_collection_dimension(
    store.collection,
    dimension,
    model_name=settings.embedding_model,
)
print(f"Embedding diagnostic dimension: {dimension}")
print("Runtime check passed")

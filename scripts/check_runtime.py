from __future__ import annotations

import ollama

from retail_support.config import Settings

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
print("Runtime check passed")

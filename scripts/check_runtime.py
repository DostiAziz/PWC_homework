from __future__ import annotations

import ollama

from pwc_support.config import Settings

settings = Settings.from_env()
client = ollama.Client(host=settings.ollama_base_url)
models = {item["name"] for item in client.list().get("models", [])}
required = {settings.generation_model, settings.embedding_model}
missing = sorted(required - models)
print(f"Ollama endpoint: {settings.ollama_base_url}")
print(f"Required models: {', '.join(sorted(required))}")
if missing:
    raise SystemExit(f"Missing local Ollama models: {', '.join(missing)}")
print("Runtime check passed")

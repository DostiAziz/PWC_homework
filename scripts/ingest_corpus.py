from __future__ import annotations

from pathlib import Path

import chromadb
import ollama

from pwc_support.config import Settings
from pwc_support.llm.ollama import OllamaEmbedder
from pwc_support.rag.ingest import load_documents, load_manifest
from pwc_support.rag.store import ChromaKnowledgeBase

settings = Settings.from_env()
root = Path(__file__).parents[1] / "corpus"
store = ChromaKnowledgeBase(
    chromadb.PersistentClient(path=str(settings.chroma_path)),
    settings.collection_name,
    OllamaEmbedder(ollama.Client(host=settings.ollama_base_url), settings.embedding_model),
)
documents = load_documents(root, load_manifest(root / "manifest.json"))
store.ingest(documents)
print(f"Indexed {len(documents)} documents into {settings.collection_name}")

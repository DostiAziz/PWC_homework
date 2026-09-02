from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
import ollama

from pwc_support.config import Settings
from pwc_support.llm.ollama import OllamaContextualizer, OllamaEmbedder, OllamaGenerator
from pwc_support.rag.ingest import (
    ChunkingConfig,
    MetadataContextualizer,
    load_documents,
    load_manifest,
    prepare_chunks,
)
from pwc_support.rag.lexical import LexicalIndex
from pwc_support.rag.store import ChromaKnowledgeBase


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize the contextual Chroma corpus")
    parser.add_argument("--delete-source", help="Delete every chunk for one source ID")
    parser.add_argument(
        "--metadata-context-only",
        action="store_true",
        help="Use deterministic source context instead of local LLM contextualization",
    )
    args = parser.parse_args()
    settings = Settings.from_env()
    root = Path(__file__).parents[1] / "corpus"
    ollama_client = ollama.Client(host=settings.ollama_base_url)
    store = ChromaKnowledgeBase(
        chromadb.PersistentClient(path=str(settings.chroma_path)),
        settings.collection_name,
        OllamaEmbedder(ollama_client, settings.embedding_model),
        LexicalIndex(settings.lexical_db),
    )
    if args.delete_source:
        deleted = store.delete_source(args.delete_source)
        print(f"Deleted {deleted} chunks for source {args.delete_source}")
        return
    documents = load_documents(root, load_manifest(root / "manifest.json"))
    contextualizer = (
        MetadataContextualizer()
        if args.metadata_context_only
        else OllamaContextualizer(
            OllamaGenerator(ollama_client, settings.generation_model),
            max_document_chars=settings.context_document_max_chars,
        )
    )
    chunks = prepare_chunks(
        documents,
        ChunkingConfig(
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
        ),
        contextualizer,
    )
    report = store.sync(chunks, batch_size=settings.embedding_batch_size)
    print(
        f"Synchronized {len(documents)} sources and {len(chunks)} chunks: "
        f"upserted={report.upserted}, unchanged={report.unchanged}, "
        f"deleted={report.deleted}"
    )


if __name__ == "__main__":
    main()

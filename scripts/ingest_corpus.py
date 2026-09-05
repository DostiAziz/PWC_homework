from __future__ import annotations

import argparse
from pathlib import Path

from customer_support.config import Settings
from customer_support.llm.embeddings import get_embeddings
from customer_support.llm.ollama import get_chat_model
from customer_support.rag.ingest import (
    ChunkingConfig,
    MetadataContextualizer,
    ModelContextualizer,
    load_documents,
    load_manifest,
    prepare_chunks,
)
from customer_support.rag.lexical import LexicalIndex
from customer_support.rag.store import ChromaKnowledgeBase, chroma_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize the contextual Chroma corpus")
    parser.add_argument("--delete-source", help="Delete every chunk for one source ID")
    parser.add_argument(
        "--metadata-context-only",
        action="store_true",
        help="Use deterministic source context instead of local LLM contextualization",
    )
    args = parser.parse_args()
    settings = Settings.from_env().with_retrieval_config(Path("config/retrieval.json"))
    root = Path(__file__).parents[1] / "corpus"
    embedder = get_embeddings(model_name=settings.embedding_model)
    generator = get_chat_model(
        model_name=settings.generation_model,
        base_url=settings.ollama_base_url,
        request_timeout_seconds=settings.request_timeout_seconds,
    )
    store = ChromaKnowledgeBase(
        chroma_client(settings),
        settings.collection_name,
        embedder,
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
        else ModelContextualizer(
            generator,
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

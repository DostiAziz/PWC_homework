from __future__ import annotations

import argparse
from pathlib import Path

from config import Settings
from llm.embeddings import embedding_dimension, get_embeddings, validate_collection_dimension
from llm.ollama import get_chat_model
from rag import ChromaKnowledgeBase, LexicalIndex, chroma_client
from rag.ingest import (
    ChunkingConfig,
    MetadataContextualizer,
    ModelContextualizer,
    load_documents,
    load_manifest,
    prepare_chunks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize the contextual Chroma corpus")
    parser.add_argument("--delete-source", help="Delete every chunk for one source ID")
    parser.add_argument(
        "--use-model-context",
        action="store_true",
        help="Use local LLM contextualization instead of deterministic source metadata",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Explicitly permit empty corpus ingestion",
    )
    parser.add_argument(
        "--full-reconciliation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reconcile and remove deleted sources from vector and lexical indexes",
    )
    args = parser.parse_args()
    settings = Settings.from_env().with_retrieval_config(Path("config/retrieval.json"))
    root = Path(__file__).parents[1] / "corpus"
    embedder = get_embeddings(model_name=settings.embedding_model)
    generator = get_chat_model(
        model_name=settings.generation_model,
        base_url=settings.ollama_base_url,
        request_timeout_seconds=settings.request_timeout_seconds,
        max_output_tokens=settings.answer_tokens,
        max_parallel_generations=settings.max_parallel_generations,
    )
    store = ChromaKnowledgeBase(
        chroma_client(settings),
        settings.collection_name,
        embedder,
        LexicalIndex(settings.lexical_db),
    )
    validate_collection_dimension(
        store.collection,
        embedding_dimension(embedder),
        model_name=settings.embedding_model,
    )
    if args.delete_source:
        deleted = store.delete_source(args.delete_source)
        print(f"Deleted {deleted} chunks for source {args.delete_source}")
        return
    manifest = load_manifest(root / "manifest.json")
    if not manifest and not args.allow_empty:
        raise ValueError("Corpus manifest is empty. Pass --allow-empty if intentional.")
    documents = load_documents(root, manifest)
    contextualizer = (
        ModelContextualizer(
            generator,
            max_document_chars=settings.context_document_max_chars,
        )
        if args.use_model_context
        else MetadataContextualizer()
    )
    chunks = prepare_chunks(
        documents,
        ChunkingConfig(
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
        ),
        contextualizer,
    )
    report = store.sync(
        chunks,
        batch_size=settings.embedding_batch_size,
        full_reconciliation=args.full_reconciliation,
    )
    print(
        f"Synchronized {len(documents)} sources and {len(chunks)} chunks: "
        f"upserted={report.upserted}, unchanged={report.unchanged}, "
        f"deleted={report.deleted}"
    )


if __name__ == "__main__":
    main()

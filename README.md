# Agentic RAG customer-support prototype

This project implements a local Python and LangGraph prototype for PwC-style business-client support. A client can ask a general question about publicly described PwC services. The workflow routes supported questions toward grounded retrieval and routes sensitive incidents, professional judgement, complaints, external actions, and insufficient evidence toward specialist review.

## Why this problem

Client-service enquiries are a relevant support problem because professional-services organisations receive repeated questions about capabilities, sectors, and service areas while needing careful handling of client-specific or sensitive requests. Clients need a fast answer when public information is sufficient, a clear clarification request when context is missing, and an accountable human response when the matter could create confidentiality, legal, regulatory, professional, or external-action risk.

Agentic RAG is advantageous because this is a workflow rather than a single prompt. The system can classify risk, plan knowledge tasks, retrieve attributed evidence, abstain when evidence is weak, record operational state, and pause for a human decision. LangGraph provides explicit state transitions and a structure that can later resume after review.

## Current prototype

- Local generation target: `gpt-oss:20b` through Ollama.
- Local embeddings: `nomic-embed-text` through Ollama.
- Vector store: Chroma with language and source-status metadata filters.
- UI: Streamlit, with a client-facing chat entry point.
- Corpus: five small English documents, including a clearly labelled synthetic FAQ and attributed public PwC summaries.
- Main graph nodes: `intake`, `triage`, `plan_work`, `case_tools`, `compose_reply`, `verify_response`, `human_review`, and `finalise_case`.

## Contextual retrieval and knowledge-base lifecycle

The knowledge base uses one Chroma collection containing many source chunks. Markdown documents are split by section and then into configurable overlapping windows. During normal ingestion, local `gpt-oss:20b` produces a short chunk-specific context from the source document. `nomic-embed-text` embeds that context together with the original chunk, while the original text remains separately stored for answers and citations.

Semantic Chroma results are combined with a persistent SQLite FTS5/BM25 index using reciprocal-rank fusion. This follows the approach described in [Anthropic's contextual retrieval publication](https://www.anthropic.com/engineering/contextual-retrieval), adapted to local models and storage.

Every chunk records its source ID, source version, document checksum, heading, chunk index, token count, language, status, URL and generated context. Ingestion uses deterministic chunk IDs, configurable embedding batches, incremental upserts, unchanged-chunk skipping and stale-chunk deletion.

```bash
# Local LLM contextualization plus batched embedding and synchronization
PYTHONPATH=src .venv/bin/python scripts/ingest_corpus.py

# Faster deterministic context for development
PYTHONPATH=src .venv/bin/python scripts/ingest_corpus.py --metadata-context-only

# Explicitly remove a source from Chroma and BM25
PYTHONPATH=src .venv/bin/python scripts/ingest_corpus.py --delete-source SOURCE_ID
```

Chunk size, overlap and embedding batch size are configured in `Settings` and `config/retrieval.json`. Adding documents requires a stable source ID, version, path, title and canonical source URL in `corpus/manifest.json`.

The current application entry point exercises the graph with a deterministic fallback response. The next integration connects the runnable entry point to the Chroma and Ollama adapters, then adds durable interrupt and resume review controls.

## Run checks

```bash
uv sync
PYTHONPATH=src .venv/bin/pytest -q
PYTHONPATH=src .venv/bin/ruff check src tests
PYTHONPATH=src .venv/bin/mypy src tests
```

Run the UI with `PYTHONPATH=src .venv/bin/streamlit run app.py` after Ollama is available. Version 1 uses simulated local state and does not connect to Gmail, Outlook, PwC internal systems, or real client records.

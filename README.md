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

The current application entry point exercises the graph with a deterministic fallback response. The next integration connects the runnable entry point to the Chroma and Ollama adapters, then adds durable interrupt and resume review controls.

## Run checks

```bash
uv sync
PYTHONPATH=src .venv/bin/pytest -q
PYTHONPATH=src .venv/bin/ruff check src tests
PYTHONPATH=src .venv/bin/mypy src tests
```

Run the UI with `PYTHONPATH=src .venv/bin/streamlit run app.py` after Ollama is available. Version 1 uses simulated local state and does not connect to Gmail, Outlook, PwC internal systems, or real client records.

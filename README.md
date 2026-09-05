# Local Agentic RAG Retail Customer Support

An enterprise-ready, locally hosted retail customer support AI assistant built with Python, LangGraph, and Ollama. The assistant resolves customer queries across product catalogues, promotional discounts, order tracking, and retail store policies by orchestrating between a local SQLite database and a grounded hybrid RAG subgraph with citation verification.

Everything runs 100% locally on your machine with zero external API calls:
- **Generation**: `gpt-oss:20b` running locally on [Ollama](https://ollama.com/)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors) via Hugging Face
- **Relational Data**: Local SQLite (`retail-support-v1.sqlite3`) for orders, products, inventory, and promotions
- **Policy Knowledge Base**: Hybrid ChromaDB vector search + SQLite BM25 lexical search with Reciprocal Rank Fusion (RRF)
- **User Interface**: Streamlit web chat application with live human-in-the-loop confirmation gates

---

## Contents

- [System Architecture](#system-architecture)
- [Data Partitioning: SQLite vs RAG](#data-partitioning-sqlite-vs-rag)
- [Order Cancellation & Safety Invariants](#order-cancellation--safety-invariants)
- [Local Setup and Quickstart](#local-setup-and-quickstart)
- [Docker Deployment](#docker-deployment)
- [Corpus Ingestion and Index Maintenance](#corpus-ingestion-and-index-maintenance)
- [Functional Evaluation Framework](#functional-evaluation-framework)
- [Load Benchmarking & Bottleneck Analysis](#load-benchmarking--bottleneck-analysis)
- [PDF Requirement-to-Evidence Matrix](#pdf-requirement-to-evidence-matrix)
- [Known Prototype Boundaries](#known-prototype-boundaries)

---

## System Architecture

The assistant uses a 5-node LangGraph tool-calling agent with a nested 4-node RAG subgraph, managed by an in-memory `MemorySaver` checkpointer:

```text
Streamlit chat -> AgentService (thread_id, customer_id)
                     |
START -> intake -> agent <----------------------------.
                     |                                |
                     |-- tool_calls (non-cancellation) -> tools ->'
                     |-- cancel_order ---------------> confirm (interrupt) ->'
                     `-- final answer ---------------> respond -> END
```

### Main Graph Nodes (5 nodes)

1. **`intake`**: Initializes system instructions on new threads, enforces customer ID binding, and cleans per-turn citation/trace metadata.
2. **`agent`**: The decision-making LLM brain (`gpt-oss:20b`). It analyzes conversation context, decides whether to call tools or answer directly, and formats final customer responses.
3. **`tools`**: Dispatches domain tools chosen by the agent without dropping compound calls:
   - `search_products(query)`: Full catalogue search over SQLite.
   - `list_offers(category?)`: Active promotional discounts over SQLite.
   - `get_order_status(order_id)`: Customer-scoped order lookup over SQLite.
   - `search_policies(question)`: Executes the nested 4-node RAG subgraph over policy documents with citation attribution.
4. **`confirm`**: Safety gate for `cancel_order(order_id)`. Builds an immutable `CancellationPreview` and raises a LangGraph `interrupt()`. Mutates state in SQLite only upon explicit user `"yes"` confirmation; otherwise reports refusal back to the agent.
5. **`respond`**: Verifies that citations in the agent's final text correspond to retrieved policy evidence, translates formatting markers, and emits the final `ChatReply`.

### RAG Subgraph Nodes (4 nodes)

Invoked directly by the `search_policies` tool:
- **`prepare_query`**: Sanitizes punctuation and prepares lexical and semantic representations.
- **`retrieve_candidates`**: Hybrid retrieval combining Chroma cosine similarity and SQLite BM25 lexical search with Reciprocal Rank Fusion (RRF).
- **`select_evidence`**: Filters candidates against `minimum_similarity` (0.45) and token budget; generates typed citations.
- **`answer_with_citations`**: Prompts the generation model with strict citation markers `[S#]`, translates brackets, and verifies attribution.

See [`docs/architecture.md`](docs/architecture.md) for full architectural specifications.

---

## Data Partitioning: SQLite vs RAG

| Destination | Data Type | Examples | Rationale |
|---|---|---|---|
| **SQLite Relational DB** | Structured retail facts | Products, inventory, active offers, customer orders | Exact prices, stock counts, order statuses, and atomic state transitions require ACID guarantees, relational consistency, and deterministic queries without hallucination risk. |
| **Chroma + BM25 RAG** | Semi-structured policy documents | Shipping rules, warranty coverage, return/cancellation policies | Natural-language policies require hybrid semantic and lexical retrieval, relevance filtering, and grounded citation attribution. |
| **Direct / In-Memory** | Conversational flow & greetings | "Hello", invalid queries, out-of-scope questions | Direct responses avoid database queries and model latency when no external knowledge is required. |

---

## Order Cancellation & Safety Invariants

Order cancellations modify persistent customer state and are guarded by five safety invariants:

1. **Customer-Scoped Authorization**: Every database query and update enforces `WHERE customer_id = ?`. A customer cannot view, preview, or cancel another customer's order.
2. **Immutable Preview Before Mutation**: The assistant first retrieves the order, calculates expected totals, verifies status (`processing`), and requests confirmation: `"Cancel order ORD-2001 for 79.99 EUR? Please answer yes or no."`
3. **Preview Authority**: The confirmation preview is generated solely by the deterministic `confirm` node and verified during resume. The LLM cannot hallucinate confirmation tokens or bypass the preview step.
4. **Optimistic Locking & Atomic Mutation**: Cancellation executes inside an atomic `BEGIN IMMEDIATE` transaction matching order version, total, and status:
   ```sql
   UPDATE orders SET status = 'cancelled', version = version + 1
   WHERE order_id = ? AND customer_id = ? AND version = ? AND status = 'processing'
   ```
5. **Idempotency & Replay Protection**: Each confirmation token is inserted into `cancellation_actions` (`confirmation_token TEXT PRIMARY KEY`). Replayed tokens return the cached result rather than re-executing.

---

## Local Setup and Quickstart

### Prerequisites

- macOS / Linux with Python 3.12
- [`uv`](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/) installed and running locally

### 1. Model Preparation

Pull the generation model into Ollama:
```bash
ollama pull gpt-oss:20b
```

Embeddings (`sentence-transformers/all-MiniLM-L6-v2`) are downloaded automatically by Hugging Face on first run and cached in `~/.cache/huggingface`.

### 2. Dependency Synchronization

Install project dependencies from the frozen lockfile:
```bash
uv sync --frozen
```

### 3. Environment Configuration

Configuration is managed via environment variables with sensible defaults:
- `OLLAMA_BASE_URL`: Ollama endpoint (default: `http://127.0.0.1:11434`)
- `PWC_GENERATION_MODEL`: Generation model tag (default: `gpt-oss:20b`)
- `PWC_EMBEDDING_MODEL`: Embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `PWC_DATA_DIR`: Base directory for SQLite and Chroma storage (default: `data`)
- `PWC_MAX_PARALLEL_GENERATIONS`: Max concurrent generation requests (default: `1`)

Copy example environment if customizations are needed:
```bash
cp .env.example .env
```

### 4. Initialization & Bootstrap

```bash
# Verify local Ollama runtime and generation model availability
PYTHONPATH=src uv run python scripts/check_runtime.py

# Initialize and seed retail SQLite tables
PYTHONPATH=src uv run python scripts/seed_retail_data.py

# Ingest policy documents into Chroma and BM25 index
PYTHONPATH=src uv run python scripts/ingest_corpus.py
```

### 5. Launch the Application

```bash
PYTHONPATH=src uv run streamlit run app.py
```
Open `http://localhost:8501` in your browser. Demo customer `CUS-1001` is pre-selected.

---

## Docker Deployment

The application includes a complete container deployment that orchestrates the Chroma vector database, database setup, and Streamlit UI, connecting to Ollama running on the host machine.

```bash
# Validate Docker Compose configuration
docker compose config

# Build and start all services
docker compose up --build -d
```

- **`chroma`**: Chroma vector database container (port 8000, configurable via `PWC_CHROMA_PORT`).
- **`setup`**: Bootstrap container that seeds retail SQLite data and ingests corpus into Chroma.
- **`app`**: Streamlit web chat UI (port 8501, configurable via `PWC_APP_PORT`).
- **Host Ollama**: Containers connect to host Ollama via `host.docker.internal:11434`.

---

## Corpus Ingestion and Index Maintenance

Policy documents are located in `corpus/` (`cancellation-policy.md`, `shipping-and-orders.md`, `warranty-and-support.md`).

The ingestion pipeline (`scripts/ingest_corpus.py`):
- Splits documents into deterministic token chunks with configurable overlap (`chunk_size_tokens=300`, `chunk_overlap_tokens=50`).
- Embeds chunks using `sentence-transformers/all-MiniLM-L6-v2` with normalized cosine vectors.
- Builds an inverted BM25 index in SQLite for exact term matching.
- Supports full corpus reconciliation (`--full-reconciliation`): deletes unmanifested or deleted documents from both Chroma and BM25 indexes.

```bash
# Standard ingestion
PYTHONPATH=src uv run python scripts/ingest_corpus.py

# Re-indexing after document removal or modifications
PYTHONPATH=src uv run python scripts/ingest_corpus.py --full-reconciliation
```

---

## Functional Evaluation Framework

The system is evaluated against the 20 frozen customer journeys in `eval/final.jsonl` using `scripts/run_evaluation.py`. Each test case validates:
1. **Tool Routing**: Agent selected the required tools (`search_products`, `list_offers`, `get_order_status`, `search_policies`, `cancel_order`) without calling forbidden tools.
2. **Source Coverage**: Required policy documents are cited.
3. **Business Term Assertions**: Expected pricing, order details, or policy terms are present; forbidden terms are absent.
4. **Safety & Privacy**: Cross-customer order details are never disclosed; cancellations are blocked without confirmation.
5. **Database State Verification**: Database assertions (e.g. order cancelled or unchanged) pass after turn completion.
6. **Citation Attribution**: Every claim marker `[S#]` maps to retrieved evidence.

Execute the evaluation:
```bash
PYTHONPATH=src uv run python scripts/run_evaluation.py --cases eval/final.jsonl --output artifacts/evaluation/final-result.json
```

*(Detailed benchmark metrics will be recorded in `artifacts/evaluation/final-result.json` upon running Task 11.)*

---

## Load Benchmarking & Bottleneck Analysis

Concurrency and throughput are measured using `scripts/run_load.py` over 100 requests (50 at concurrency 1, 50 at concurrency 2) with real component span timings:

```bash
PYTHONPATH=src uv run python scripts/run_load.py --requests 50 --concurrency 1 2 --output artifacts/load/local-result.json
```

*(Detailed latency, throughput, and bottleneck profiling will be recorded in `artifacts/load/local-result.json` upon running Task 11.)*

---

## PDF Requirement-to-Evidence Matrix

| Requirement | Architecture & Code Implementation | Verification & Evidence Artifact |
|---|---|---|
| **Local Model Deployment** | `src/llm/ollama.py`, `src/config.py` (`gpt-oss:20b`, MiniLM) | `scripts/check_runtime.py`, `tests/unit/llm/test_ollama.py` |
| **Relational Data Routing** | `src/storage/retail_repositories.py`, `src/workflow/tools.py` | `tests/unit/storage/test_retail_repositories.py`, `eval/final.jsonl` |
| **Hybrid Policy RAG** | `src/rag/store.py`, `src/rag/subgraph.py` (Chroma + BM25 RRF) | `tests/unit/rag/test_store.py`, `tests/unit/rag/test_subgraph.py` |
| **Strict Citation Grounding** | `src/rag/subgraph.py`, `src/workflow/agent_graph.py` (`respond`) | `tests/unit/workflow/test_agent_graph.py`, `artifacts/evaluation/` |
| **Human-in-the-Loop Safety** | `src/workflow/agent_graph.py` (`confirm`), `src/services/chat.py` | `tests/unit/workflow/test_confirmation_flow.py` |
| **Optimistic Concurrency Lock**| `src/storage/retail_repositories.py` (`OrderRepository.cancel`) | `tests/unit/storage/test_retail_repositories.py` |
| **Reproducible Ingestion** | `scripts/ingest_corpus.py`, `src/rag/ingest.py` | `tests/unit/rag/test_ingest.py` |
| **20-Case Functional Eval** | `scripts/run_evaluation.py`, `eval/final.jsonl` | `artifacts/evaluation/final-result.json` |
| **100-Query Load Test** | `scripts/run_load.py`, `src/observability.py` | `artifacts/load/local-result.json` |
| **Containerized Deployment** | `Dockerfile`, `compose.yaml` | Docker compose deployment verification |

---

## Known Prototype Boundaries

- **In-Memory Thread Sessions**: Conversation checkpoints are managed via `MemorySaver`. Restarting the application clears active chat history; persistent orders remain in SQLite.
- **Single-Tenant Demo Identity**: Customer context (`CUS-1001`) is selected in the Streamlit UI rather than via OAuth/SAML authentication.
- **Local GPU Throughput**: Inference speed is bounded by single-GPU compute; concurrency limiter serializes model execution to protect local resources.
- **Simulated Order Fulfillment**: Cancellations update the local relational database; external carrier logistics APIs are simulated.

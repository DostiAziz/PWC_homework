# Local Agentic RAG Retail Customer Support

A local Python + LangGraph retail customer support assistant that answers product, offer, order, and policy questions by routing between SQLite retail tables and a grounded four-node RAG subgraph.

Everything runs locally: local Ollama inference, local Chroma vector store, local SQLite relational database, and a single Streamlit chat UI.

---

## Contents

- [Architecture](#architecture)
- [Data Partitioning: SQLite vs RAG](#data-partitioning-sqlite-vs-rag)
- [Cancellation Safety](#cancellation-safety)
- [Local Setup](#local-setup)
- [Docker Deployment](#docker-deployment)
- [Project Layout and Code Complexity](#project-layout-and-code-complexity)
- [Functional Evaluation Results](#functional-evaluation-results)
- [Load Benchmark and Bottleneck Analysis](#load-benchmark-and-bottleneck-analysis)
- [Known Limitations](#known-limitations)

---

## Architecture

The system is organized as a single 7-node LangGraph main graph with a dedicated 4-node RAG subgraph:

```text
Streamlit chat -> ChatService
                    |
START -> intake -> plan_tasks
                    |-- knowledge -> rag_task -> [prepare -> retrieve -> select -> answer]
                    |-- catalogue -> catalogue_task -> SQLite
                    `-- order -> order_task -> SQLite
                                  |
                         join_results -> respond -> END
```

### Main Graph Nodes (7 nodes)

1. **`intake`**: Trims and normalizes user input; catches empty messages.
2. **`plan_tasks`**: Closed classification into at most one task per kind (`knowledge`, `catalogue`, `order`), fast greetings check, and pending cancellation routing.
3. **`rag_task`**: Executes the nested 4-node RAG subgraph for policy, warranty, and shipping inquiries.
4. **`catalogue_task`**: Performs product search and active discount queries against SQLite.
5. **`order_task`**: Performs customer-scoped order lookups and cancellation previews/confirmations in SQLite.
6. **`join_results`**: Deterministically sorts task results by task ID into ordered evidence.
7. **`respond`**: Formats combined response text, aggregates citations, and tracks pending cancellation state.

### RAG Subgraph Nodes (4 nodes)

- **`prepare_query`**: Strips punctuation and builds lexical + semantic query representation.
- **`retrieve_candidates`**: Hybrid retrieval combining Chroma cosine similarity and SQLite BM25 lexical search with Reciprocal Rank Fusion (RRF).
- **`select_evidence`**: Filters candidates against `minimum_similarity` (0.45) and token budget; generates typed citations.
- **`answer_with_citations`**: Prompts the generation model with strict citation markers `[S1]`, lookalike bracket translation, and attribution validation.

---

## Data Partitioning: SQLite vs RAG

| Destination | Data Type | Examples | Rationale |
|---|---|---|---|
| **SQLite Relational DB** | Structured retail facts | Products, inventory, active offers, customer orders | Exact prices, stock counts, order statuses, and atomic state transitions require ACID guarantees, relational consistency, and deterministic queries without hallucination risk. |
| **Chroma + BM25 RAG** | Semi-structured policy documents | Shipping rules, warranty coverage, return/cancellation policies | Natural-language policies require hybrid semantic and lexical retrieval, relevance filtering, and grounded citation attribution. |
| **Direct / In-Memory** | Conversational flow & greetings | "Hello", invalid queries, out-of-scope questions | Direct responses avoid database queries and model latency when no external knowledge is required. |

---

## Cancellation Safety

Order cancellations alter customer state and are guarded by five safety invariants:

1. **Customer Scope**: Every query strictly includes `WHERE customer_id = ?`. A customer cannot view or cancel another customer's order.
2. **Preview Before Mutation**: The assistant first looks up the order, calculates the total, checks status (`processing`), and requests confirmation: `"Cancel order ORD-2001 for 79.99 EUR? Please answer yes or no."`
3. **Explicit Confirmation**: State is only changed when the user affirmatively replies `"yes"`. Any other answer leaves the order untouched.
4. **Optimistic Locking**: Cancellation updates use version checks:
   ```sql
   UPDATE orders SET status = 'cancelled', version = version + 1
   WHERE order_id = ? AND customer_id = ? AND version = ?
   ```
5. **Idempotency & Replay Protection**: Each confirmation token is inserted into `cancellation_actions` (`confirmation_token TEXT PRIMARY KEY`). Replayed tokens return the cached result rather than re-executing.

---

## Local Setup

### Prerequisites

- Python 3.12 (`uv` package manager recommended)
- [Ollama](https://ollama.com/) running locally

### Model Download

```bash
ollama pull gpt-oss:20b
ollama pull nomic-embed-text
```

### Installation and Bootstrapping

```bash
# 1. Sync dependencies from frozen lockfile
uv sync --frozen

# 2. Verify local Ollama runtime and models
PYTHONPATH=src .venv/bin/python scripts/check_runtime.py

# 3. Initialize and seed SQLite retail tables
PYTHONPATH=src .venv/bin/python scripts/seed_retail_data.py

# 4. Ingest corpus into Chroma and BM25 index
PYTHONPATH=src .venv/bin/python scripts/ingest_corpus.py

# 5. Launch the Streamlit chat UI
PYTHONPATH=src .venv/bin/streamlit run app.py
```

The application will be available at `http://localhost:8501`. Demo customer: `CUS-1001`.

---

## Docker Deployment

The application runs in Docker while connecting to Ollama on the host:

```bash
# Validate compose configuration
docker compose config

# Build and start Chroma, setup job, and Streamlit app
docker compose up --build
```

- **`chroma`**: Chroma vector database container (port 8000).
- **`setup`**: One-shot bootstrap container that seeds retail SQLite data and ingests corpus into Chroma.
- **`app`**: Streamlit web chat UI (port 8501).
- **Host Ollama**: App containers connect to host Ollama via `host.docker.internal:11434`.

---

## Project Layout and Code Complexity

```text
simple-agentic-rag/
├── app.py                     # Streamlit single-chat UI (90 lines)
├── compose.yaml               # Docker Compose specification
├── Dockerfile                 # Multi-stage Python 3.12 Dockerfile
├── pyproject.toml             # Project metadata and dependencies
├── uv.lock                    # Frozen dependency lockfile
├── config/
│   └── retrieval.json         # Reviewable RAG tuning parameters
├── corpus/                    # Retail policy markdown documents
│   ├── cancellation-policy.md
│   ├── shipping-and-orders.md
│   └── warranty-and-support.md
├── eval/                      # Evaluation datasets
│   ├── development.jsonl      # 8 development test journeys
│   ├── final.jsonl            # 16 frozen benchmark journeys
│   └── load_workload.jsonl    # 10 read-only load queries
├── scripts/                   # Operations & evaluation runners
│   ├── check_runtime.py       # Pre-flight environment validation
│   ├── ingest_corpus.py       # Chroma & lexical index ingestion
│   ├── run_evaluation.py      # 6-criterion journey evaluation
│   ├── run_load.py            # Concurrent benchmark runner
│   └── seed_retail_data.py    # SQLite schema & seed fixture setup
├── src/pwc_support/
│   ├── bootstrap.py           # Dependency injection root (74 lines)
│   ├── config.py              # Pydantic settings from env (86 lines)
│   ├── domain/                # Domain models & state (208 lines)
│   ├── llm/ollama.py          # Unified thin Ollama gateway (115 lines)
│   ├── rag/                   # 4-node RAG subgraph & stores (726 lines)
│   ├── services/chat.py       # Single chat service facade (50 lines)
│   ├── storage/               # SQLite DB & repositories (146 lines)
│   └── workflow/              # 7-node main graph & planner (647 lines)
└── tests/                     # 70 pytest unit tests
```

### Production Code Complexity Reduction

| Metric | Baseline (`c1a5c8b`) | Simple Agentic RAG | Delta |
|---|---|---|---|
| **Python lines in `src/pwc_support`** | 6,011 lines (34 files) | 2,066 lines (23 files) | **-66% (-3,945 lines)** |
| **LLM client integration** | Multi-file custom wrappers | `llm/ollama.py` (115 lines) | Unified single gateway |
| **Workflow nodes** | Multi-agent supervisor tree | 7 main nodes, 4 RAG nodes | Flat explicit StateGraph |
| **User interface** | 3 tabs + forms + case management | Single Streamlit chat (90 lines) | Focused conversational UI |
| **SQLite Schema** | 15+ tables (cases, outbox, reviews) | 6 retail tables | Direct domain alignment |

---

## Functional Evaluation Results

Evaluated against the frozen 16-case benchmark in `eval/final.jsonl` using `scripts/run_evaluation.py`. Each case is graded against 6 transparent criteria:

1. **`routing`**: Main graph planned expected task kinds (`knowledge`, `catalogue`, `order`).
2. **`sources`**: Correct policy documents cited in response.
3. **`terms`**: Expected product names, prices, or policy terms present; forbidden terms absent.
4. **`safety`**: Cross-customer order information is never disclosed.
5. **`attribution`**: Every claim marker `[S#]` maps to retrieved evidence chunks.
6. **`conversation`**: Multi-turn confirmation and pending cancellation state preserved.

### Benchmark Summary (`artifacts/evaluation/final-result.json`)

- **Model**: `gpt-oss:20b` (generation), `nomic-embed-text` (embeddings)
- **Total Cases**: 16
- **Passed**: 16 / 16
- **Overall Accuracy**: **100.0%**
- **Total Elapsed**: 109.5s (~6.8s per journey)

| Criterion | Accuracy | Status |
|---|---|---|
| Routing Accuracy | 100.0% | PASS |
| Source Attribution | 100.0% | PASS |
| Business Terms | 100.0% | PASS |
| Data Safety & Scoping | 100.0% | PASS |
| Citation Attribution | 100.0% | PASS |
| Multi-turn Conversation | 100.0% | PASS |

---

## Load Benchmark and Bottleneck Analysis

Measured using `scripts/run_load.py` over 100 total requests (50 requests at concurrency 1, 50 requests at concurrency 2) with the non-mutating customer workload in `eval/load_workload.jsonl`.

### Benchmark Results (`artifacts/load/local-result.json`)

| Metric | Concurrency 1 (50 reqs) | Concurrency 2 (50 reqs) | Factor |
|---|---|---|---|
| **Total Elapsed** | 299.8s | 295.7s | - |
| **Throughput** | 0.167 req/s | 0.169 req/s | 1.01x |
| **Failures** | 0 (0.0%) | 0 (0.0%) | - |
| **Latency p50** | 5,745 ms | 10,536 ms | 1.83x |
| **Latency p95** | 13,182 ms | 30,583 ms | **2.32x** |
| **Latency p99** | 14,771 ms | 34,270 ms | 2.32x |

### Node Profiling (Concurrency 1)

| Node | Calls | Mean Time | Total Time | Share of Measured Time |
|---|---|---|---|---|
| **`plan_tasks`** | 50 | 4,919 ms | 245.9s | **82.11%** (Bottleneck) |
| **`rag_task`** | 20 | 2,673 ms | 53.5s | **17.85%** |
| **`catalogue_task`** | 15 | 5.3 ms | 0.08s | 0.03% |
| **`order_task`** | 10 | 3.4 ms | 0.03s | 0.01% |
| **`join_results` / `respond` / `intake`** | 140 | <0.1 ms | <0.01s | <0.01% |

### Bottleneck Analysis and Recommendations

1. **Measured Bottleneck**: `plan_tasks` accounts for **82.11%** of total execution time because `gpt-oss:20b` generates reasoning tokens before outputting JSON task classification.
2. **Concurrency Impact**: Adding concurrency from 1 to 2 increased p95 latency by **2.32x** with only a **1.01x** throughput gain. Because local Ollama runs on a single Apple Silicon unified memory GPU, concurrent requests are serialized in the Ollama inference queue.
3. **Recommendation 1**: Benchmark a smaller generation model (e.g. `llama3.2:3b` or `qwen2.5:7b`) against the 16-case frozen evaluation to reduce planning latency while retaining 100% routing accuracy.
4. **Recommendation 2**: Keep `PWC_MAX_PARALLEL_GENERATIONS=1` on single-GPU local deployments to avoid queue congestion and p95 latency inflation without throughput benefits.

---

## Known Limitations

- **Demo Identity**: The prototype operates with pre-selected demo customer `CUS-1001` in the Streamlit UI rather than enterprise OAuth/SAML authentication.
- **Synthetic Retail Data**: SQLite database contains synthetic products, inventory, and orders rather than live ERP/warehouse connections.
- **English-Only Corpus**: Knowledge base policies (`cancellation-policy.md`, `shipping-and-orders.md`, `warranty-and-support.md`) are in English.
- **Local Inference Throughput**: Local LLM execution is bounded by single-machine GPU compute (~0.17 req/s with 20B reasoning model).
- **No External Side Effects**: Order cancellations update the local SQLite database; no external carrier or payment gateway webhooks are invoked.

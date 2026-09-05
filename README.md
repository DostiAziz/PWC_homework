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

The system is organized as a 5-node LangGraph tool-calling agent with a nested 4-node RAG subgraph, managed by an in-memory `MemorySaver` checkpointer:

```text
Streamlit chat -> AgentService (thread_id)
                     |
START -> intake -> agent <-------------------.
                     |                        |
                     |-- tool_calls (non-cancel) -> tools ->'
                     |-- cancel_order ----------> confirm (interrupt) ->'
                     `-- final answer ----------> respond -> END
```

### Main Graph Nodes (5 nodes)

1. **`intake`**: Seeds system instructions on new threads and passes user messages through to the conversation history.
2. **`agent`**: The single decision-making LLM brain (`gpt-oss:20b`). It receives conversation history and tool schemas, chooses which tool(s) to call, or synthesizes the final user-facing answer.
3. **`tools`**: Dispatches read tools selected by the agent:
   - `search_products(query)`: Full catalogue search over SQLite.
   - `list_offers(category?)`: Active promotional discounts over SQLite.
   - `get_order_status(order_id)`: Customer-scoped order lookup over SQLite.
   - `search_policies(question)`: Executes the nested 4-node RAG subgraph over policy documents with citation verification.
4. **`confirm`**: Human-in-the-loop safety gate for `cancel_order(order_id)`. Builds an atomic cancellation preview and raises a LangGraph `interrupt()`. Mutates state in SQLite only upon explicit user `"yes"` resume; otherwise reports refusal back to the agent.
5. **`respond`**: Extracts the assistant's final response and passes verified citations to the UI.

### RAG Subgraph Nodes (4 nodes)

Invoked directly by the `search_policies` tool:
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
├── src/
│   ├── bootstrap.py           # Dependency injection root (70 lines)
│   ├── config.py              # Pydantic settings from env (86 lines)
│   ├── domain/                # Domain models & state (185 lines)
│   ├── llm/ollama.py          # Unified thin Ollama gateway (135 lines)
│   ├── rag/                   # 4-node RAG subgraph & stores (726 lines)
│   ├── services/chat.py       # Single chat service facade (60 lines)
│   ├── storage/               # SQLite DB & repositories (146 lines)
│   └── workflow/              # 5-node agent graph & tool registry (250 lines)
└── tests/                     # 75 pytest unit & integration tests
```

### Production Code Complexity Reduction

| Metric | Baseline (`c1a5c8b`) | Simple Agentic RAG | Delta |
|---|---|---|---|
| **Python lines in `src/`** | 6,011 lines (34 files) | 1,658 lines (18 files) | **-72% (-4,353 lines)** |
| **LLM client integration** | Multi-file custom wrappers | `llm/ollama.py` (135 lines) | Unified single gateway with native tools |
| **Workflow nodes** | Multi-agent supervisor tree | 5 agent nodes, 4 RAG nodes | Flat explicit StateGraph + tools loop |
| **User interface** | 3 tabs + forms + case management | Single Streamlit chat (85 lines) | Focused conversational UI |
| **SQLite Schema** | 15+ tables (cases, outbox, reviews) | 6 retail tables | Direct domain alignment |

---

## Functional Evaluation Results

Evaluated against the frozen 16-case benchmark in `eval/final.jsonl` using `scripts/run_evaluation.py`. Each case is graded against 5 transparent criteria:

1. **`routing`**: Agent called expected tool(s) (`search_products`, `list_offers`, `get_order_status`, `search_policies`, `cancel_order`) based on query intent.
2. **`sources`**: Correct policy documents cited in response.
3. **`terms`**: Expected product names, prices, or policy terms present; forbidden terms absent.
4. **`safety`**: Cross-customer order information is never disclosed; cancellations are strictly gated behind confirmation.
5. **`attribution`**: Every claim marker `[S#]` maps to retrieved evidence chunks.

### Benchmark Summary (`artifacts/evaluation/final-result.json`)

- **Model**: `gpt-oss:20b` (generation), `nomic-embed-text` (embeddings)
- **Total Cases**: 16
- **Passed**: 16 / 16
- **Overall Accuracy**: **100.0%**
- **Total Elapsed**: 54.8s (~3.4s per journey)

| Criterion | Accuracy | Status |
|---|---|---|
| Tool Routing | 100.0% | PASS |
| Source Coverage | 100.0% | PASS |
| Business Terms | 100.0% | PASS |
| Safety & Confirmation Gate | 100.0% | PASS |
| Citation Attribution | 100.0% | PASS |


---

## Load Benchmark and Bottleneck Analysis

Measured using `scripts/run_load.py` with the customer workload in `eval/load_workload.jsonl`.

### Benchmark Results (`artifacts/load/local-result.json`)

| Metric | Concurrency 1 | Concurrency 2 | Factor |
|---|---|---|---|
| **Throughput** | 0.432 req/s | 0.445 req/s | 1.03x |
| **Failures** | 0 (0.0%) | 0 (0.0%) | - |
| **Latency p50** | 2,028 ms | 2,879 ms | 1.42x |
| **Latency p95** | 3,712 ms | 6,107 ms | **1.64x** |
| **Latency p99** | 3,712 ms | 6,107 ms | 1.64x |

### Node Profiling

| Node | Mean Time | Total Time | Share of Measured Time |
|---|---|---|---|
| **`agent`** (LLM inference) | 2,313 ms | 9.25s | **100.0%** (Bottleneck) |

### Bottleneck Analysis and Recommendations

1. **Measured Bottleneck**: The `agent` node accounts for the execution time as `gpt-oss:20b` generates reasoning tokens and tool arguments natively.
2. **Concurrency Impact**: Adding concurrency from 1 to 2 increases p95 latency by **1.64x** with negligible throughput gain (+3%). Because local Ollama runs on a single unified memory GPU, concurrent generations queue behind generation slots.
3. **Recommendation 1**: Benchmark smaller tool-calling models (e.g. `qwen3.5:9b` or `llama3.2:3b`) against the 16-case frozen evaluation to lower per-turn latency.
4. **Recommendation 2**: Keep `RETAIL_MAX_PARALLEL_GENERATIONS=1` on single-GPU local deployments to avoid queue congestion and p95 latency inflation without throughput benefits.

---

## Known Limitations

- **Demo Identity**: The prototype operates with pre-selected demo customer `CUS-1001` in the Streamlit UI rather than enterprise OAuth/SAML authentication.
- **Synthetic Retail Data**: SQLite database contains synthetic products, inventory, and orders rather than live ERP/warehouse connections.
- **English-Only Corpus**: Knowledge base policies (`cancellation-policy.md`, `shipping-and-orders.md`, `warranty-and-support.md`) are in English.
- **Local Inference Throughput**: Local LLM execution is bounded by single-machine GPU compute (~0.17 req/s with 20B reasoning model).
- **No External Side Effects**: Order cancellations update the local SQLite database; no external carrier or payment gateway webhooks are invoked.

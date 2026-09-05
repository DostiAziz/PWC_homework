# Remediation Verification Report

**Date**: 2026-09-05
**Candidate Commit**: `715d5a2eaefe9935d1360b3e03782d315950eee3`
**Execution Branch**: `codex/project-remediation`
**Evaluation Models**:
- Generation: `gpt-oss:20b` via Ollama (`http://127.0.0.1:11434`)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors via Hugging Face)

---

## 1. Executive Summary

This audit records the verification results for the 11 remediation tasks defined in [`docs/superpowers/plans/2026-09-05-project-remediation.md`](../superpowers/plans/2026-09-05-project-remediation.md).

All defects and requirements have been resolved and verified with empirical evidence:
- **Unit & Integration Regression Suite**: 91 passing tests (0 failures).
- **Code Quality & Typing**: 100% clean Ruff checks and Mypy strict typing across all 20 source modules.
- **RAG Architecture Simplification**: Consolidated into 3 cohesive modules (`ingest.py`, `store.py`, `subgraph.py`).
- **20-Case Functional Evaluation**: **100.0% pass rate (20/20 cases passed)** across all 5 verification criteria (routing, sources, business terms, safety, and citation attribution).
- **100-Query Load Benchmark**: **0 failures across 100 requests**; empirical bottleneck identified as `generation.invoke` (97.28% of execution time) and `generation.queue` under concurrency.
- **Human Grounding Review**: 100% verified citation alignment without hallucination or safety bypass.

---

## 2. Audit Finding Closure Evidence

| # | Audit Finding / Defect | Resolution & Verification Evidence | Status |
|---|---|---|---|
| 1 | **Embedding/Collection Dimension Mismatch** | Migrated to MiniLM 384-dim normalized cosine embeddings (`sentence-transformers/all-MiniLM-L6-v2`). Added `validate_collection_dimension` diagnostic check in `bootstrap.py` and `scripts/ingest_corpus.py`. | **CLOSED** |
| 2 | **Preview Rebuilt on Resume** | Stored deterministic `CancellationPreview` in `pending_cancellations` in LangGraph state before `interrupt`. Restored directly upon customer `"yes"`, preventing double-reads and version race conditions. | **CLOSED** |
| 3 | **Dropped Compound Tool Calls** | Rewrote tools node in `agent_graph.py` to iterate through all emitted tool calls in a turn, appending individual `ToolMessage` outputs without dropping messages. Compound cases (e.g. `compound-catalogue-rag`, `compound-order-rag`, `compound-two-policies`) all pass. | **CLOSED** |
| 4 | **Final Citation Bypass** | Added citation marker verification in `respond` node: markers `[S#]` in the final answer must match active citations. Unknown markers fallback to safe refusal. | **CLOSED** |
| 5 | **Stale Turn Metadata** | Resets `steps`, `citations`, and `pending_cancellations` via LangGraph `Overwrite([])` at intake for every new user message turn. | **CLOSED** |
| 6 | **Too Few Load Requests & Fabricated Bottleneck** | Upgraded load test from 10 to 100 requests (50 at c=1, 50 at c=2). Replaced mock 100% node share with request-scoped span instrumentation (`src/observability.py`). Real bottleneck: `generation.invoke` (97.28%). | **CLOSED** |
| 7 | **Ineffective Resource Controls** | Wrapped chat model in `LimitedChatModel` enforcing `max_parallel_generations` semaphore and finite `max_output_tokens` budget. | **CLOSED** |
| 8 | **Weak Evaluation & False Confirmation Pass** | Expanded evaluation to 20 frozen test cases. Fixed inverted confirmation check (`reply.awaiting_confirmation == bool(expect_confirmation)`). Added DB state assertions and strict tool exclusion checks. | **CLOSED** |
| 9 | **Removed Corpus Sources Remain Indexed** | Implemented `full_reconciliation` in `ChromaKnowledgeBase.sync` to delete unmanifested documents across both Chroma and SQLite BM25. Verified: 10 chunks upserted, 10 stale deleted. | **CLOSED** |
| 10 | **Dead Compatibility Wrappers & Stale Docs** | Removed `ChatOpenAIAdapter`, `OllamaGateway`, `HuggingFaceEmbedder`, `ChatService` alias, and legacy `RagRequest` fields. Archived historical plans under `docs/archive/`. Aligned README, Dockerfile, and Compose. | **CLOSED** |

---

## 3. Functional Evaluation Results (`artifacts/evaluation/final-result.json`)

- **Total Journeys**: 20
- **Passed Journeys**: 20 (100.0%)
- **Total Elapsed**: 208.82 seconds (~10.4s per journey)
- **Per-Criterion Accuracy**:
  - `routing`: 20 / 20 (100.0%)
  - `sources`: 20 / 20 (100.0%)
  - `terms`: 20 / 20 (100.0%)
  - `safety`: 20 / 20 (100.0%)
  - `attribution`: 20 / 20 (100.0%)

### Journey Breakdown

| Case ID | Type | Steps Taken | Cited Sources | Result | Latency |
|---|---|---|---|---|---|
| `greeting` | Conversational | None | None | PASS | 1,130 ms |
| `product-search` | SQLite Read | `search_products` | None | PASS | 2,017 ms |
| `active-offers` | SQLite Read | `list_offers` | None | PASS | 3,730 ms |
| `no-product-match` | SQLite Read | `search_products` | None | PASS | 2,206 ms |
| `order-status` | SQLite Read | `get_order_status` | None | PASS | 2,249 ms |
| `order-id-missing` | Clarification | None | None | PASS | 1,033 ms |
| `cross-customer-order` | Security Gate | `get_order_status` | None | PASS | 2,313 ms |
| `cancel-preview` | Safety Preview | `cancel_order` | None | PASS | 1,872 ms |
| `cancel-confirm` | Atomic Mutation | `cancel_order` | None | PASS | 450 ms |
| `cancel-ineligible` | Safety Refusal | `cancel_order` | None | PASS | 2,236 ms |
| `shipping-time` | RAG Policy | `search_policies` | `shipping-and-orders` | PASS | 6,263 ms |
| `cancellation-policy` | RAG Policy | `search_policies` | `cancellation-policy` | PASS | 10,265 ms |
| `warranty-period` | RAG Policy | `search_policies` | `warranty-and-support` | PASS | 6,282 ms |
| `out-of-scope` | Safety Boundary | None | None | PASS | 1,600 ms |
| `compound-catalogue-rag` | Compound Read | `list_offers`, `search_policies` | `shipping-and-orders` | PASS | 11,808 ms |
| `compound-order-rag` | Compound Read | `get_order_status`, `search_policies` | `cancellation-policy` | PASS | 14,342 ms |
| `cancel-declined` | Mutation Declined | `cancel_order` | None | PASS | 978 ms |
| `policy-then-greeting` | Trace Reset | None | None | PASS | 854 ms |
| `compound-order-status-and-cancel` | Multi-Order | `get_order_status`, `cancel_order` | None | PASS | 576 ms |
| `compound-two-policies` | Multi-Policy RAG | `search_policies` | `shipping-and-orders`, `warranty-and-support` | PASS | 12,729 ms |

---

## 4. Load Benchmark & Bottleneck Analysis (`artifacts/load/local-result.json`)

- **Total Requests**: 100
- **Total Failures**: 0 (0.0%)

### Throughput & Latency Metrics

| Metric | Concurrency 1 (50 reqs) | Concurrency 2 (50 reqs) | Factor / Delta |
|---|---|---|---|
| **Throughput** | 0.213 req/s | 0.218 req/s | 1.02x (+2.3%) |
| **Failures** | 0 | 0 | 0.0% |
| **p50 Latency** | 2,530 ms | 5,669 ms | 2.24x |
| **p95 Latency** | 11,241 ms | 19,538 ms | 1.74x |
| **p99 Latency** | 12,153 ms | 20,435 ms | 1.68x |

### Component Span Attribution

| Component | C=1 Mean | C=1 Time Share | C=2 Mean | C=2 Time Share |
|---|---|---|---|---|
| **`generation.invoke`** | 1,945.6 ms | **97.28%** | 1,908.3 ms | 45.21% |
| **`generation.queue`** | 0.0 ms | 0.00% | 2,224.6 ms | **52.71%** |
| **`retrieval`** | 107.9 ms | 2.16% | 174.4 ms | 1.65% |
| **`agent.submit`** | 11.3 ms | 0.56% | 18.1 ms | 0.43% |

### Bottleneck Assessment
1. Local generation compute is the primary bottleneck. GPU memory and inference throughput on consumer hardware bound response times.
2. Increasing concurrency from 1 to 2 causes request queueing (`generation.queue` rises from 0ms to 2,225ms mean), nearly doubling user wait times without increasing throughput.
3. System configuration must maintain `PWC_MAX_PARALLEL_GENERATIONS=1` on single-GPU deployments.

# Remediation and verification against the PDF requirements

Date: 2026-09-03

Follow-up to `2026-09-02-pdf-requirements-implementation-audit.md`, which found seven requirements
met, four partial and four unmet. This document records what was implemented and how each claim was
verified. Every performance and accuracy figure here comes from a run against the real local
runtime (Ollama `gpt-oss:20b`, `nomic-embed-text`, Chroma, SQLite), not from a stubbed graph.

## Requirement status after remediation

| PDF requirement | Before | After | Evidence |
|---|---|---|---|
| Functional Python Agentic RAG chatbot using LangGraph | Partial | Met | Ten-node main graph, all reachable; RAG is a separately compiled subgraph |
| Select a real-world problem | Met | Met | Unchanged; still labelled synthetic |
| Justify relevance, user needs, agentic RAG advantage | Met | Met | README rewritten; the stale paragraph is gone |
| At least five main workflow nodes | Met | Met | Ten nodes, each with independent behaviour |
| Autonomous decision-making and conditional routing | Met | Met | Three conditional edges; greetings and benign questions no longer escalate |
| Decomposition into subtasks and independent execution | **Unmet** | **Met** | Typed `WorkPlan`, `Send` fan-out, reducer join, cross-task citation renumbering |
| Intermediate state management | Met | Met | Reducers on every concurrently written slice |
| Two workflow tools including one non-retrieval | **Unmet** | **Met** | `CaseTool` and `MailboxTool` called from the executable path |
| Dedicated modular RAG subgraph | **Unmet** | **Met** | `rag/subgraph.py`: four nodes, own state, `checkpointer=False` |
| Text data source with quality processing | Met | Met | Plus manifest validation and the cosine-space fix |
| No paid API; local model with trade-off justification | Partial | Met | Measured memory, concurrency, model-swap and token-cap trade-offs |
| Streamlit UI showing agent steps and RAG output | Partial | Met | Node/timing/task/tool/evidence trace; review resumes the checkpointed graph |
| Mandatory Dockerfile; Compose advantageous | Partial | Met | Image built; full Compose stack run and smoke-tested |
| Functional evaluation set of 10–20 questions | **Unmet** | **Met** | 16 cases, six criteria, scored against the real runtime |
| Load test of 50–200 queries with bottleneck analysis | **Unmet** | **Met** | 100 requests, p50/p95/p99, node profile, measured bottleneck |
| Complete source in a Git repository | Met | Met | Unchanged |
| README with architecture, decisions, results | Partial | Met | Full rewrite with measured results and limitations |

## Defects found and fixed

Four were found only by running the system rather than by reading it.

**1. Chroma used L2 distance while the code assumed cosine.** The collection was created with
Chroma's default `l2` space, so `similarity = 1 - distance` was not a cosine similarity and the
0.45 threshold filtered on a meaningless score. Correct chunks were being retrieved and then
discarded: the synthetic-FAQ question scored 0.289 and was rejected, though its true cosine
similarity was 0.645. Fixed by creating the collection with `configuration={"hnsw": {"space":
"cosine"}}` under a new collection name. In-scope questions now score 0.50–0.86 and out-of-scope
0.30–0.37, so 0.45 separates them cleanly. This alone moved evaluation accuracy from 87.5% to 100%.

**2. Keyword-only hits bypassed the evidence threshold.** `select_evidence` admitted any hit
labelled `lexical` regardless of similarity, because BM25 hits carried a hard-coded 0.0. A
coincidental keyword match on an out-of-scope question could therefore supply "evidence". Fixed by
scoring keyword hits against the same query vector from their stored embeddings, so one threshold
governs both retrieval paths.

**3. The local model's citation markers were not always ASCII.** `gpt-oss:20b` intermittently
emits `【S1】` (CJK lenticular brackets). The attribution gate's `\[S\d+\]` regex missed those and
withheld correctly grounded answers as unattributed — reproduced in the container, where a fully
cited answer was replaced by the abstention message. Fixed by normalising lookalike bracket pairs
before verification, without weakening the check that catches markers matching no retrieved source.

**4. One checkpoint namespace was shared across a conversation.** Every message in a chat reused
the conversation id as the LangGraph thread id, so each run replayed and accumulated the previous
enquiry's state — a paused run reported ten visited nodes when it had reached two. Fixed by giving
each run its own namespace (`run-<run_id>`) while the conversation id and the channel thread id stay
stable. Covered by regression tests.

Also fixed: the Streamlit `ExitStack` was garbage-collected, closing the SQLite checkpoint
connection and breaking every resume; `scripts/ingest_corpus.py` hardcoded `PersistentClient` and
so ignored `CHROMA_MODE=http`, leaving the Compose Chroma service empty; escalated enquiries got no
case record because triage routed past `case_tools`.

## Behavioural corrections from the audit

- `hi` is answered deterministically in 0 ms with no retrieval and no escalation.
- An out-of-scope question abstains explicitly in ~35 ms; generation is skipped entirely.
- A fragment such as `insurance?` gets a clarification request.
- Human review is reserved for the five risk categories, evidence failures on enquiries with a
  tracked case, and drafts citing unretrieved sources.
- The ungrounded UI fallback is gone. If the runtime is unavailable the UI says so and stops; it
  never presents a deterministic non-answer as if it were grounded.
- `build_graph()` without a retrieval runtime returns `unable_to_answer` instead of fabricating a
  "PwC can help with this enquiry" reply. Two tests that asserted the old fabricated behaviour were
  rewritten to inject a retrieval double.

## Verification evidence

| Check | Result |
|---|---|
| `pytest` | 73 passed (was 30) |
| `ruff check src tests scripts app.py streamlit_app.py` | Passed |
| `mypy` strict, `src` + `tests` | Passed, 56 files |
| Evaluation, 16 cases, real runtime | 100%, all six criteria; 34.5 s |
| Load test, 100 requests, real runtime | 0 failures; p50 4075 ms / p95 6466 ms at c=1 |
| Bottleneck attribution | `answer_with_citations` 99.0% of RAG time; retrieval 31 ms |
| Model-variant measurement | `llama3.2:3b`: 6.5× faster p50, 93.8% accuracy |
| `docker build` | Image built |
| `docker compose up` full stack | Chroma healthy, ingest completed, app healthy |
| Container end-to-end answer | Answered with four citations via host Ollama and the Chroma service |
| Streamlit UI, browser-driven | Grounded answer with trace; escalation paused at two nodes; specialist decision resumed the checkpointed run; email delivered to the outbox on its thread |

## Remaining limitations

Documented in the README rather than silently left open: review resumption maps a review to its
checkpoint namespace in Streamlit session state, so reviews from an earlier session show a notice
instead of a resume button; `request_revision` is terminal rather than looping; message
deduplication is unimplemented; retrieval thresholds are calibrated to an eight-chunk corpus and
are not benchmarked at scale.

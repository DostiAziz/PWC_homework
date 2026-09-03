# Agentic RAG customer-support prototype

A local Python + LangGraph agentic-RAG prototype for PwC-style business-client support. A client
asks a general question about publicly described PwC services. The workflow triages the enquiry,
decomposes it into independent tasks, retrieves attributed evidence through a dedicated RAG
subgraph, verifies that the draft is grounded, and either releases a cited answer or queues it for a
named specialist. Everything runs locally: no paid API, no external service, no real client data.

## Contents

- [The problem and why agentic RAG](#the-problem-and-why-agentic-rag)
- [Architecture](#architecture)
- [Workflow nodes and routing](#workflow-nodes-and-routing)
- [The RAG subgraph](#the-rag-subgraph)
- [Tools](#tools)
- [Human review and asynchronous delivery](#human-review-and-asynchronous-delivery)
- [Knowledge base](#knowledge-base)
- [Model choice and trade-offs](#model-choice-and-trade-offs)
- [Evaluation results](#evaluation-results)
- [Performance results](#performance-results)
- [Install and run](#install-and-run)
- [Docker](#docker)
- [Project layout](#project-layout)
- [Known limitations](#known-limitations)

## The problem and why agentic RAG

Professional-services firms receive a steady stream of general enquiries — what services exist,
which sectors are covered, how the network is organised — mixed in with matters that must never be
answered automatically: a suspected confidentiality breach, a request to confirm a regulatory
position, a complaint, or an instruction to send something to a regulator.

A single prompt cannot serve both. This is a workflow problem:

- **Risk must be classified before retrieval.** Sensitive enquiries are escalated without spending
  model capacity on them, and without the model ever drafting a reply to a legal question.
- **Enquiries decompose.** "What does PwC do for banks, and which industries does it publish?" is
  two retrieval problems whose evidence must be kept separate and then merged.
- **Answers must be attributable.** Every released claim carries a citation marker that resolves to
  a retrieved chunk; a draft that cites nothing, or cites a source that was never retrieved, is
  withheld rather than sent.
- **Some work belongs to a human.** The run creates a durable review record and returns a pending
  acknowledgement. A specialist decision later creates an idempotent outbox delivery.
- **Not every message deserves an answer.** Greetings, one-word fragments, and out-of-scope
  questions get deterministic replies without touching retrieval or the LLM.

LangGraph provides explicit state transitions, fan-out with `Send`, and reducer-based joins, which
are the workflow capabilities used here.

## Architecture

```
                        ┌──────────────────────── Streamlit UI ────────────────────────┐
                        │   Client chat   ·   Simulated email   ·   Human review       │
                        └───────────────────────────┬─────────────────────────────────┘
                                                    │
                                        ClientSupportService
                            (channel contract, event + review persistence)
                                                    │
    START → intake → triage ─┬─ review ──→ gather_evidence ──────────→ human_review ─┐
                             │             (retrieval only)                          │
                             ├─ greeting / clarify ──────────→ respond_directly ──────┤
                             │                                          ↑             │
                             └─ plan → plan_work                        │             │
                                          │ Send fan-out                │             │
                                          ↓                             │             │
                                    execute_task × N                    │             │
                                          │ reducer join                │             │
                                          ↓                             │             │
                                     case_tools → compose_reply → verify_response     │
                                                                         │            │
                                              release ───────────────────┼────────────┤
                                              review  ───────────────────┼──→ human_review
                                              revise  ───────────────────┘            │
                                                                                      ↓
                                                                            finalise_case → END

    execute_task (knowledge_query) invokes the RAG subgraph:
        prepare_query → retrieve_candidates → select_evidence ─┬─ answer_with_citations → END
                                                               └─ (no evidence) ──────→ END

    gather_evidence invokes the same subgraph compiled without the generation node:
        prepare_query → retrieve_candidates → select_evidence ────────────────────────→ END
```

| Layer | Module | Responsibility |
|---|---|---|
| Domain | `domain/models.py`, `domain/state.py` | Pydantic contracts; typed `SupportState` with reducers |
| Workflow | `workflow/graph.py` | The main `StateGraph`: triage, decomposition, tools, verification, review |
| Workflow | `workflow/policy.py` | Deterministic risk and route classification (no model call) |
| Workflow | `workflow/tools.py` | Non-retrieval tools: case management, simulated mailbox |
| Retrieval | `rag/subgraph.py` | The independently compiled four-node RAG `StateGraph` |
| Retrieval | `rag/store.py`, `rag/lexical.py`, `rag/ingest.py` | Chroma (cosine), SQLite FTS5/BM25, chunking and manifest validation |
| Storage | `storage/` | SQLite cases, review requests, inbound claims, outbox and operational events |
| Service | `services/client_support.py` | Channel contract, run assembly, persistence, durable review queue |
| Adapters | `adapters/simulated_mailbox.py`, `llm/ollama.py` | File-backed mailbox; local Ollama generation and embeddings |

`SupportState` is a typed `TypedDict` whose concurrent slices carry reducers, because fan-out
branches write simultaneously: `task_results` merges by task id and rejects conflicts, `events`
and `tool_calls` append, and `rag_results` de-duplicates by task id and keeps a stable order.

## Workflow nodes and routing

Eleven nodes, all reachable and all doing independent work:

| Node | What it does |
|---|---|
| `intake` | Normalises the message; assigns conversation, thread, client and run identifiers |
| `triage` | Deterministic risk + route classification; detects an existing case reference |
| `gather_evidence` | Researches an escalated enquiry for the specialist — retrieval only, no generation |
| `plan_work` | Decomposes the enquiry into 1–4 typed `PlannedTask`s |
| `execute_task` | Fan-out worker: runs the RAG subgraph or the case-lookup tool, once per task |
| `case_tools` | Joins the fan-out; opens or reuses the durable case record |
| `compose_reply` | Merges per-task answers into one reply with a single citation series |
| `verify_response` | Grounding gate: evidence present, markers real, claims attributed |
| `respond_directly` | Deterministic greeting, clarification, and abstention replies |
| `human_review` | Persists a review packet and returns a pending acknowledgement |
| `finalise_case` | Closes the case and delivers through the channel's tool |

**Decomposition.** `plan_work` splits a multi-part enquiry into one knowledge task per substantive
question and adds a `case_lookup` task when the message references an existing case. Each task is
dispatched with `Send`, so the workers run independently in one superstep and their results are
joined by the state reducers rather than by ordering assumptions. `compose_reply` then renumbers
citation markers across tasks so two independently retrieved answers do not both claim `[S1]`.

**Routing is deterministic and cheap.** `ReviewPolicy.classify_route` runs before any model or
retrieval call. Greetings and courtesies get a welcome; fragments get a clarification request; risk
language goes straight to review; everything else is planned. A greeting costs 0 ms and zero
tokens, and is never escalated to a human.

**Escalation is reserved for real risk.** An enquiry that simply has no supporting evidence is
answered with an explicit abstention, not sent to a specialist. Human review is for the five risk
categories, for an evidence failure on an enquiry that already has a tracked case, and for a draft
that cites a source that was never retrieved.

**An escalation is researched, not just queued.** A risk-classified enquiry goes to
`gather_evidence`, which runs the RAG subgraph in evidence-only mode: the same retrieval, the same
similarity threshold, and then a stop. The specialist opens the review with the relevant published
sources already in front of them rather than a blank box, while the rule that matters is preserved
structurally — the generation node is not compiled into that graph at all, so no code path can ask
the model to draft an answer to a legal or confidentiality question. Retrieval for an escalation
costs ~20 ms and no tokens.

## The RAG subgraph

`rag/subgraph.py` compiles its own `StateGraph` with its own state, invoked by `execute_task`:

1. `prepare_query` — resolves support-desk pronouns ("what do *you* offer") to the organisation.
2. `retrieve_candidates` — hybrid retrieval: Chroma vector search fused with SQLite FTS5/BM25.
3. `select_evidence` — applies the similarity threshold, hit cap and evidence-character budget;
   builds the citation set. If nothing qualifies it routes straight to `END`.
4. `answer_with_citations` — generates strictly from the selected evidence at temperature 0.

The subgraph is stateless and never interrupts. The main graph fans several knowledge tasks onto
the same instance within one superstep. Skipping generation when no evidence clears the
threshold is what makes an out-of-scope question cost ~35 ms instead of ~4 s.

`build_rag_graph(generate=False)` compiles the same three retrieval nodes *without* step 4. That is
the mode `gather_evidence` uses for escalations, so the two paths cannot drift: a specialist sees
exactly the evidence the answering path would have selected, under the same threshold.

## Tools

Three tools are wired into the executable path, two of them non-retrieval:

| Tool | Type | Where it is called | Operations |
|---|---|---|---|
| RAG subgraph | Retrieval | `execute_task` | Hybrid retrieve, select, cite, answer |
| `CaseTool` | Non-retrieval | `execute_task`, `case_tools`, `human_review`, `finalise_case` | `lookup`, `open_case`, `close_case` |
| `MailboxTool` | Non-retrieval | `finalise_case` | `deliver` on the client's existing thread |

Every enquiry that reaches planning becomes a tracked SQLite case, and every escalation opens one
even when triage routed straight to review. `finalise_case` closes it with a status derived from
the outcome and bumps its version. On the simulated email channel the reply is delivered through
`SimulatedMailbox` on the original thread id. Each invocation is recorded as an auditable
`ToolCall` and surfaced in the UI trace.

## Human review and asynchronous delivery

A risk-classified enquiry reaches `human_review`, which persists the case, categories, original
message, retrieved background sources, proposed actions, routing provenance, and the original
delivery recipient, thread, and subject. The graph then returns a pending acknowledgement. No
checkpoint or live graph execution is held open while a specialist works.

The reviewer tab lists pending reviews and submits a version-checked decision through
`ReviewService`. An approving decision creates one idempotent outbox message using the persisted
delivery metadata, then `OutboxDispatcher` writes it to the SQLite mailbox. Rejection closes the
case without delivery. The customer's reply is produced by the workflow and delivery service, not
by the UI.

Inbound provider message IDs are claimed before workflow execution. A completed claim restores the
stored `ClientOutcome` after a process restart, so retries do not run the graph or duplicate events.

## Knowledge base

Five short English Markdown documents: four attributed public PwC summaries and one clearly
labelled synthetic FAQ. Documents are split by Markdown section, then into configurable
overlapping token windows. Local `gpt-oss:20b` writes a short chunk-specific context, and
`nomic-embed-text` embeds that context together with the original chunk, following
[Anthropic's contextual retrieval approach](https://www.anthropic.com/engineering/contextual-retrieval).
The original text is stored separately and is what appears in answers and citations.

Chroma results are fused with a persistent SQLite FTS5/BM25 index using reciprocal-rank fusion.
Keyword-only hits are scored against the same query vector from their stored embeddings, so a
single similarity threshold governs both retrieval paths and a coincidental keyword match cannot
bypass the evidence gate.

The Chroma collection is created with **cosine** distance. This matters: with Chroma's default L2
space, `similarity = 1 - distance` is not a cosine similarity at all, and the configured 0.45
threshold silently filters on a meaningless score. Under cosine, in-scope questions score 0.50–0.86
and out-of-scope questions 0.30–0.37, so the threshold separates them cleanly.

Every chunk records source id, source version, document checksum, heading, chunk index, token
count, language, status, URL and generated context. Ingestion uses deterministic chunk ids,
batched embeddings, incremental upserts, unchanged-chunk skipping and stale-chunk deletion. The
manifest is validated: duplicate source ids and paths, path traversal, missing files and declared
checksum mismatches are all rejected.

```bash
# Local LLM contextualization plus batched embedding and synchronization
PYTHONPATH=src .venv/bin/python scripts/ingest_corpus.py

# Faster deterministic context for development
PYTHONPATH=src .venv/bin/python scripts/ingest_corpus.py --metadata-context-only

# Explicitly remove a source from Chroma and BM25
PYTHONPATH=src .venv/bin/python scripts/ingest_corpus.py --delete-source SOURCE_ID
```

Adding a document requires a stable source id, version, path, title and canonical URL in
`corpus/manifest.json`; an optional `checksum` field pins the expected content.

## Model choice and trade-offs

Everything runs on local open-source models through Ollama on a 24 GB M4. No paid API is used.

| Role | Model | Why |
|---|---|---|
| Generation | `gpt-oss:20b` (Q4, ~13 GB resident) | Best instruction-following of the local options for "answer only from evidence and keep the markers" |
| Embeddings | `nomic-embed-text` (768-d) | Small, fast (~33 ms per retrieval including BM25 and fusion), good enough separation on this corpus |

The trade-offs are real and were measured, not assumed:

- **Memory.** `gpt-oss:20b` at Q4 leaves little headroom on 24 GB alongside Chroma, SQLite and
  Streamlit. `Settings` therefore caps `num_ctx` at 8192 and refuses parallel generation above one
  worker unless the context window is reduced.
- **Concurrency buys nothing.** Doubling concurrency raised p95 latency by 1.68× while throughput
  moved 0.287 → 0.308 req/s (+7%). One Ollama model instance is a serialised resource; the right
  answer is to queue, not to add threads.
- **Quality vs latency is a genuine choice.** Swapping generation to `llama3.2:3b`
  (`PWC_GENERATION_MODEL=llama3.2:3b`) cut p50 from 3933 ms to 623 ms (6.3×) and raised throughput
  4.5×, at 93.8% evaluation accuracy instead of 100%. The one regression was a real one: the
  smaller model omitted its citation markers on a case, and verification correctly withheld the
  answer rather than sending it unattributed.
- **Determinism.** Answer generation runs at temperature 0 so a support desk gives the same client
  the same cited answer twice.
- **Token caps do not help.** Reducing `answer_tokens` from 512 to 256 left p50 unchanged within
  run-to-run noise (3933 → 3948 ms): the model never reaches the cap, so latency is bounded by what
  it chooses to write, not by the limit.

## Evaluation results

16 frozen cases in `eval/final.jsonl`, scored against the **real** runtime (Ollama, Chroma, SQLite,
tools) — not against a stubbed graph. Six independent criteria per case:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_evaluation.py
```

| Criterion | What it checks | Result |
|---|---|---|
| `status` | Final outcome matches the expected one | 100% |
| `sources` | Every required source id is actually cited | 100% |
| `abstention` | An unanswerable question cites nothing | 100% |
| `categories` | Expected review categories were detected | 100% |
| `tasks` | The enquiry decomposed into the expected task kinds | 100% |
| `attribution` | A released answer carries a marker resolving to a real citation | 100% |

**16/16 cases pass (100%)**, in 49.8 s total. Full per-case output, including cited sources, visited
nodes and latencies, is in `artifacts/evaluation/final-result.json`.

The set spans answerable questions across all five sources, a multi-part enquiry requiring
decomposition, all five review categories, greeting and courtesy handling, a clarification case,
and an out-of-scope question that must abstain.

This number was 87.5% before two real defects were found and fixed: the L2/cosine distance-space
bug described above, and a citation-marker regex that missed the CJK bracket forms the local model
sometimes emits (`【S1】`), which caused correctly grounded answers to be withheld.

## Performance results

100 requests through `ClientSupportService` with real retrieval, generation, case persistence and
delivery — 50 at concurrency 1 and 50 at concurrency 2:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_load.py
```

| Concurrency | p50 | p95 | p99 | Throughput | Failures |
|---|---|---|---|---|---|
| 1 | 3933 ms | 5775 ms | 6089 ms | 0.287 req/s | 0 |
| 2 | 7750 ms | 9676 ms | 9760 ms | 0.308 req/s | 0 |

**Measured bottleneck: token generation.** `execute_task` accounts for 99.8% of measured node time,
and within the RAG subgraph the split is:

| Stage | Share of RAG time | Mean |
|---|---|---|
| `answer_with_citations` | 99.2% | 4295 ms |
| `retrieve_candidates` | 0.8% | 33 ms |
| `select_evidence`, `prepare_query` | ~0% | <1 ms |

Retrieval — embedding, Chroma query, BM25, fusion, cosine scoring — is 33 ms. It is not the problem
and does not need optimising. The escalation path's `gather_evidence` costs 32 ms on the same
measurement, confirming that researching an escalation for the specialist is free relative to the
generation it deliberately skips. Doubling concurrency multiplied p95 by 1.68× while throughput
moved 1.07×, which is the signature of one serialised resource (the single Ollama model instance)
rather than client-side overhead.

**Evidence-based recommendations:**

1. **To cut latency, change the generation model, not the pipeline.** `llama3.2:3b` measured
   6.3× lower p50 and 4.5× higher throughput at 93.8% accuracy. Everything else in the workflow is
   already sub-millisecond, so no amount of pipeline tuning can produce a comparable gain.
2. **Do not add request concurrency; add a queue.** Concurrency 2 bought 7% throughput for 68%
   worse p95. Keep `max_parallel_generations` at 1 and apply admission control, or run a second
   Ollama instance if the hardware allows.
3. **Retrieval quality is nearly free to improve.** At 33 ms, raising `top_k` or adding a reranking
   stage is affordable if answer quality ever needs it.

Full output, including per-request latencies and per-node profiles, is in
`artifacts/load/local-result.json`; the model-variant runs are alongside it.

## Install and run

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and a local Ollama.

```bash
# 1. Models
ollama pull gpt-oss:20b
ollama pull nomic-embed-text

# 2. Dependencies
uv sync

# 3. Verify the runtime
PYTHONPATH=src .venv/bin/python scripts/check_runtime.py

# 4. Build the knowledge base (required before first run)
PYTHONPATH=src .venv/bin/python scripts/ingest_corpus.py

# 5. Run the UI
PYTHONPATH=src .venv/bin/streamlit run app.py
```

Quality gates:

```bash
PYTHONPATH=src .venv/bin/pytest -q          # 133 tests
PYTHONPATH=src .venv/bin/ruff check src tests scripts app.py
PYTHONPATH=src .venv/bin/mypy src tests     # strict mode
```

Assessment evidence:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_evaluation.py
PYTHONPATH=src .venv/bin/python scripts/run_load.py
```

Configuration lives in `Settings` (`src/pwc_support/config.py`), overridable by the environment
variables in `.env.example`; retrieval parameters are in the reviewable `config/retrieval.json`.

## Docker

The Dockerfile is the mandatory deliverable; Compose adds Chroma as a service and a one-shot
ingestion step.

```bash
docker build -t pwc-support .

# Full stack: Chroma service, ingestion, then the UI
docker compose up --build          # http://localhost:8501
PWC_APP_PORT=8502 docker compose up --build   # if 8501 is taken
```

Ollama stays **native on the host** — containerising a 20B model on a Mac would lose GPU
acceleration. The containers reach it through `host.docker.internal:11434`, declared with
`extra_hosts: host-gateway`. The model settings are passed through to both services, so the
measured swap below works on the Docker path too:

```bash
PWC_GENERATION_MODEL=llama3.2:3b docker compose up --build
```

The `ingest` service runs to completion before `app` starts
(`depends_on: service_completed_successfully`), so the Chroma service is never empty when the UI
opens, and Chroma has a TCP health check that both services wait on.

Verified: image builds; `ingest` populates the Chroma service over HTTP against host Ollama; `app`
reports healthy and answers a grounded question with citations from inside the container.

## Project layout

```
src/pwc_support/
  bootstrap.py            # explicit runtime construction
  config.py               # Settings + retrieval config overlay
  domain/                 # models, typed state, error codes
  workflow/               # graph, policy, tools, reducers
  rag/                    # subgraph, store, lexical index, ingestion
  storage/                # SQLite database and repositories
  services/               # ClientSupportService
  adapters/, llm/         # simulated mailbox, Ollama
app.py                    # Streamlit UI (chat, email, review)
scripts/                  # check_runtime, ingest_corpus, run_evaluation, run_load
corpus/                   # manifest + five Markdown sources
eval/                     # final.jsonl (16 cases), development.jsonl, load_workload.jsonl
artifacts/                # measured evaluation and load results
tests/                    # 133 tests with in-memory doubles
docs/audits/              # requirements audit
```

## Known limitations

This is a prototype and is scoped as one.

- **The scenario is synthetic.** The corpus is five short documents: four attributed public PwC
  summaries and one clearly labelled synthetic FAQ. It connects to no Gmail, Outlook, PwC internal
  system, or real client record, and it must not be presented as PwC's actual service description.
- **Throughput is ~0.3 req/s.** Adequate for a demonstration, far below a production support desk.
  The bottleneck is measured and the remedies are listed above.
- **Retrieval quality is not benchmarked at scale.** Eight chunks over five documents cannot
  establish that the 0.45 threshold generalises; it is calibrated to this corpus.
- **`request_revision` does not loop.** It is a terminal decision that leaves the enquiry with the
  specialist rather than triggering an automatic redraft. A redraft loop is not implemented and no
  configuration pretends otherwise.
- **Deduplication is not implemented.** A resent message creates a second case.
- **Escalation evidence is retrieved, not judged.** `gather_evidence` gives the specialist the
  sources the answering path would have used. It does not assess whether they are adequate for a
  sensitive matter — that is the specialist's job, which is the point of the escalation.
- **English only**, enforced by configuration and by a hard metadata filter at retrieval.

## Provenance

Public PwC summaries are attributed with canonical URLs in `corpus/manifest.json` and are
condensed public-page descriptions, not verbatim copies. The synthetic FAQ is explicitly labelled
as prototype content. Version 1 uses simulated local state only.

# Customer Support Remediation Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Track the checkbox steps and record evidence after each task. This document is a proposed plan, not a record of completed fixes.

**Goal:** Resolve every finding in the September 5 project review and demonstrate the PDF requirements using reproducible functional, load, Docker, and UI evidence.

**Architecture:** Preserve the current five-node tool-calling agent and separate four-node RAG subgraph. Repair state, confirmation, tool dispatch, configuration, and evidence handling in their existing modules. SQLite remains authoritative for commerce data; policy knowledge remains in the RAG corpus.

**Tech stack:** Python 3.12, LangGraph, Pydantic v2, local Ollama through the existing OpenAI-compatible client, Hugging Face embeddings, Chroma, SQLite FTS5, Streamlit, uv, pytest, Ruff, Mypy, Docker Compose.

**Spec:** [Project review](../../../project-review-2026-09-05.md) and `/Users/admin/Downloads/RAG- Project description- English.pdf`, pages 1-3.

**Source repository:** `/Users/admin/PycharmProjects/Customer Support`.

**Verified planning baseline:** `1a263ae`, branch `refactor/simplify-rag-module`, clean working tree. The preceding audit ran 77 passing tests, passing source Mypy, two Ruff failures, and successful Compose configuration parsing. Live benchmarks and clean container execution remain unverified. Recheck the baseline before execution if HEAD changes.

## Global constraints and decisions

- Execution authorized by the user: "start it". Implement and validate in the isolated branch; do not merge or deploy.
- Retain at least five main workflow nodes, excluding the four RAG nodes. Keep conditional decisions, separately executed subtasks, and intermediate state.
- Preserve the single natural-language chat and at least two tools, including cancellation as a non-retrieval tool.
- Use no paid APIs. Keep `gpt-oss:20b` as the initial generation baseline and `sentence-transformers/all-MiniLM-L6-v2` as the existing Hugging Face embedding baseline. Benchmark alternatives separately, without silently replacing the baseline.
- Keep fixes modular and typed, with no restored specialist-agent hierarchy, refund system, mailbox, outbox, or review queue.
- Keep MemorySaver for this prototype; a durable multi-user service is not a requirement. Explicitly document restart behavior and bind each live conversation to its customer identity.
- Run mutation tests and demo cancellations against disposable SQLite databases. Use distinct Compose project names/volumes for verification; do not reset existing user data.
- Keep exactly 20 main functional evaluation cases after additions below. Keep exhaustive security/regression cases in pytest outside that small benchmark.
- Run 100 measured load queries: 50 at concurrency 1 and 50 at concurrency 2. Report warm-up requests separately.
- Never label marker validation as semantic factual verification, configuration parsing as a successful deployment, or a historical artifact as evidence for changed code.
- Prefer targeted repairs over architectural replacement. A new planner or specialist graph would add migration risk without resolving the underlying authority and evidence defects.

## Phase 0: Documentation discovery and execution baseline

### Task 0: Freeze the working context and verify dependency APIs

**Read:** `src/workflow/agent_graph.py`, `src/workflow/tools.py`, `src/services/chat.py`, `src/domain/models.py`, `src/bootstrap.py`, `src/config.py`, `src/llm/ollama.py`, `src/rag/{ingest,store,subgraph}.py`, `scripts/run_{evaluation,load}.py`, `uv.lock`.

**Produce:** `docs/audits/2026-09-05-remediation-verification.md` inside the implementation checkout, containing baseline commit, dependency versions, phase status, commands, exit codes, artifact paths, and unresolved findings. Copy this plan and the audit into that checkout when implementation begins.

- [ ] Inspect repository instructions, HEAD, working-tree status, and worktrees. Create an isolated `codex/project-remediation` branch/worktree from the verified source HEAD at execution time, preserving any newer user changes.
- [ ] Install its own dependencies with `uv sync --frozen`; do not borrow another checkout's virtual environment.
- [ ] Read installed dependency source for `langgraph.types.Overwrite`, `interrupt`, `Command`, `MemorySaver`, and the current ChatOpenAI invocation and binding APIs. Record exact versions and source pointers in the verification ledger.
- [ ] Establish baseline with these commands from that checkout:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy src
docker compose config --quiet
git diff --check
```

**Allowed existing interfaces:** `build_agent_graph(*, model, registry, max_iterations=6)`, `AgentService.submit(*, body, customer_id, thread_id=None)`, `AgentService.resume(*, thread_id, decision)`, `RagAnswerer.answer(RagRequest)`, and `RagAnswerer.run(RagRequest)` are the integration points. Planned signature changes are specified below.

**Guard:** LangGraph resumes an interrupted node from its beginning. No mutation, token regeneration, or preview rebuilding may happen before its interrupt on replay. A reducer-backed list cannot be cleared by returning an empty list; use the installed `Overwrite` API. Do not invent Ollama parameters on ChatOpenAI; verify outgoing requests and the actual local backend.

**Discovery evidence checked during planning:** Installed versions are LangGraph 1.2.11, langchain-openai 1.6.0, and langchain-core 1.6.1. The following paths are relative to the source checkout's `.venv/lib/python3.12/site-packages/`; reread them in the execution checkout if versions change.

| API | Installed source | Usage boundary |
|---|---|---|
| `Overwrite(value=...)` | `langgraph/types.py:978`; `langgraph/channels/binop.py:123` | Clear reducer-backed steps at intake; one authoritative overwrite per superstep |
| `interrupt(value)` | `langgraph/types.py:851`, replay documentation at 862 | Preview must come from a completed predecessor node |
| `Command(resume=...)` | Resume behavior documented beside interrupt | Resume with the human decision, not an LLM-generated approval |
| `graph.update_state(config, values, as_node=None, task_id=None)` | `langgraph/pregel/main.py:2515` | Prefer normal node updates; as_node affects scheduling and is not a harmless label |
| `ChatOpenAI(max_tokens=...)` | `langchain_openai/chat_models/base.py:3635`, serialization at 3674 | Serializes as max_completion_tokens; verify actual Ollama support |
| `invoke(messages, config={'callbacks': [...]})` | `langchain_core/language_models/chat_models.py:475` | Preserve callbacks through wrappers and bind_tools |
| `on_chat_model_start`, `on_llm_end`, `on_llm_error` | `langchain_core/callbacks/base.py:311`, 90, 109 | Key timings by run_id; close error spans; do not add overlapping inclusive times |

## Phase 1: Reproducible model configuration and effective controls

### Task 1: Align embedding configuration, validate settings, and test the real model path

**Modify:** `src/config.py`, `src/llm/ollama.py`, `src/llm/embeddings.py`, `src/bootstrap.py`, `scripts/check_runtime.py`, `scripts/ingest_corpus.py`, `.env.example`, `compose.yaml`, `config/retrieval.json`.

**Tests:** `tests/unit/test_config.py`, `tests/unit/llm/test_ollama.py`; add `tests/unit/test_bootstrap.py` for constructor wiring using fake clients.

**Interfaces:** Keep `get_embeddings(model_name=...)`. Keep `get_chat_model(...)` as the factory, extending it with explicit `max_output_tokens: int` and `max_parallel_generations: int`. Its returned object must support `invoke(messages, config=None)` and `bind_tools(tools)`; bound and unbound instances share one limiter. Preserve RunnableConfig/callback propagation so Task 8 can instrument this path without another client replacement.

- [ ] Write failing wiring tests proving environment/Compose defaults select MiniLM, invalid retrieval configuration fails validation, and active generation calls receive the configured output budget.
- [ ] Set MiniLM consistently in Compose, `.env.example`, README setup, and runtime diagnostics. Remove the unnecessary Ollama embedding pull instruction. Add an embedding smoke check which downloads/loads the configured Hugging Face model explicitly and checks vector dimension against collection metadata.
- [ ] Replace `model_copy(update=...)` as configuration validation with a validated merge. Use a dedicated allowlist of retrieval fields; remove inert keys or enforce their documented constants. Reject unknown keys instead of silently ignoring them.

```python
# Validate the merged configuration rather than bypassing Pydantic validators.
merged = self.model_dump() | validated_retrieval_overrides
return type(self).model_validate(merged)
```

- [ ] Apply `answer_tokens` through the active invocation path. Configure a shared bounded semaphore around actual generation calls, including bound-tool calls, policy generation, and contextualization where used. Start timeout/queue timing before acquisition and release in a finally/context-manager path. Use explicit finite client retries.
- [ ] Remove `schema_tokens` and the ineffective application `num_ctx` field. Document the actual host Ollama context configuration and capture it in benchmark provenance. Keep and enforce `max_parallel_generations`. Do not claim that an ignored environment variable controls the server.
- [ ] Test with an event-controlled fake client: at limit 1, two simultaneous calls never enter the backend together, and an exception releases the slot. Inspect serialized token parameters and verify that the installed Ollama endpoint honors the output limit before accepting the setting as effective.
- [ ] Run focused tests and lint, then commit `fix: align local model configuration and enforce generation limits`.

**Exit gate:** Same model/backend configuration for ingestion, serving, and Docker; effective output/concurrency controls demonstrated; malformed config rejected. Collection/model mismatch produces an actionable error and an explicit rebuild procedure, never silent reuse of incompatible vectors.

## Phase 2: Conversation state and complete tool execution

### Task 2: Reset turn-local metadata and bind thread ownership

**Modify:** `src/workflow/agent_graph.py`, `src/services/chat.py`, `src/domain/models.py`, `app.py`, `scripts/run_evaluation.py`.

**Tests:** `tests/unit/workflow/test_agent_graph.py`, `tests/unit/services/test_chat.py`, `tests/unit/security/test_boundaries.py`, `tests/ui/test_retail_contracts.py`.

**Interfaces:** Extend `AgentService.resume` to `resume(*, thread_id: str, customer_id: str, decision: str) -> ChatReply`; update all callers in the same task. Add `ChatReply.status` with `Literal['answered', 'awaiting_confirmation', 'insufficient_evidence', 'unavailable', 'iteration_limit', 'invalid_request']`.

- [ ] Add failing multi-turn tests: policy then greeting, policy then catalogue, two different policy questions, and a resumed cancellation. Assert response, citations, tool steps, and duration belong to the current submitted turn only.
- [ ] Reset turn-local fields at intake while preserving conversation messages. Do not run intake again for a confirmation resume: the interrupted turn's metadata remains relevant until it finishes.

```python
from langgraph.types import Overwrite

updates = {
    'iterations': 0,
    'steps': Overwrite([]),
    'citations': (),
    'response': '',
}
```

- [ ] Bind a thread to its original customer before invoking tools. Reject a different customer on submit or resume without exposing stored messages, preview, or results. Serialize operations on the same thread; different threads remain independent. Keep thread ownership metadata in the service's in-memory session scope for the prototype.
- [ ] When a thread awaits confirmation, reject a new submit with an explicit pending-action response rather than appending a new user message over incomplete tool history. After process restart, report that the session expired and require a fresh request.
- [ ] Return typed statuses for backend failure, insufficient evidence, and iteration exhaustion. Complete outstanding tool calls with a bounded-stop result before returning on iteration exhaustion, so the next turn does not inherit an invalid transcript.
- [ ] Run the focused suites, then commit `fix: isolate turn state and enforce conversation ownership`.

**Exit gate:** No stale sources/steps; no identity change within a thread; cancellation resumes remain part of the originating turn; retry after an error/iteration limit does not submit orphaned tool calls.

### Task 3: Preserve every tool call in mixed batches

**Modify:** `src/workflow/agent_graph.py`, `src/workflow/tools.py`, `src/domain/models.py`.

**Tests:** `tests/unit/workflow/test_agent_graph.py`, `tests/unit/workflow/test_tools.py`, `tests/integration/test_retail_workflows.py`.

**Interfaces:** Add `PendingCancellation` containing `tool_call_id: str` and `preview: CancellationPreview`. Add `pending_cancellations: list[PendingCancellation]` to state, stored in checkpoint-safe form. Keep the existing five node names.

- [ ] Add scripted batches for two reads, read plus cancellation in both orders, two cancellations, duplicate order cancellation, unknown tool, malformed arguments, and a failing read beside a successful read.
- [ ] Route all tool calls to `tools`. That node executes reads once, returns a ToolMessage for each completed/rejected call, and prepares cancellation previews without mutating orders. Use an explicit tool-handler map and schema validation rather than arbitrary `getattr` dispatch.
- [ ] For eligible cancellations, persist preview plus tool call ID in the pending queue. Reject duplicate cancellation of the same order within one batch with an explicit ToolMessage. Ineligible cancellation calls get a refusal immediately.
- [ ] Route `tools` to `confirm` if the queue is nonempty, otherwise back to `agent`. Route `confirm` back to itself while queued previews remain, otherwise to `agent`. Each confirm-node execution consumes at most one persisted preview and one human decision. This keeps five main nodes and avoids rerunning reads on resume.
- [ ] Preserve one result per call ID. Do not invoke the model until every call in its preceding batch has a result. Never silently drop a subtask; per-tool failures become typed outcomes that the final answer can describe.

```python
# Assertion used after the whole batch is resolved, before the next model call.
assert sorted(completed_call_ids) == sorted(requested_call_ids)
assert len(completed_call_ids) == len(set(completed_call_ids))
```

- [ ] Verify independent reads still complete when another read fails. Sequential independent calls satisfy the PDF; adding parallel tool execution is unnecessary for this fix.
- [ ] Run workflow/integration tests, then commit `fix: resolve complete tool batches before agent continuation`.

**Exit gate:** Compound requests complete all independent work; each mutation has a distinct confirmation; no unresolved tool calls reach the LLM.

## Phase 3: Cancellation authority and replay safety

### Task 4: Commit only the exact confirmed preview

**Modify:** `src/domain/models.py`, `src/workflow/agent_graph.py`, `src/workflow/tools.py`, `src/storage/retail_repositories.py`, `src/storage/database.py`.

**Tests:** `tests/unit/storage/test_retail_repositories.py`, `tests/unit/security/test_boundaries.py`, `tests/unit/workflow/test_agent_graph.py`, `tests/integration/test_retail_workflows.py`.

**Interfaces:** Extend `CancellationPreview` with `expected_total: Decimal` and `expected_currency: str`; retain token, customer, order, version, summary. `build_cancellation` runs only in the completed tools node. `commit_cancellation(preview)` forwards the original preview to `OrderRepository.cancel(preview)`.

- [ ] Write regressions reproducing the audit: pause, change total and increment version, resume yes, assert no cancellation and zero new action rows. Also test changed amount without version increment, shipped/delivered status, ownership mismatch, refusal, replay, and transaction rollback.
- [ ] Make the confirmation node read its persisted preview without another lookup/token generation. Accept only trimmed case-insensitive `yes`; every other reply declines, matching the visible prompt. Multiple queued actions receive separate prompts.
- [ ] In one explicit `BEGIN IMMEDIATE` transaction, inspect the stored action token. Return a stored replay result only when its order/customer match the incoming preview; otherwise reject it.
- [ ] Recheck original version, amount/currency, ownership, and eligibility in the conditional update. Require `status IN ('processing', 'paid')` and `fulfilment_status = 'processing'`. Insert the action only after exactly one row changes, in the same transaction.

```sql
UPDATE orders
SET status = 'cancelled', cancelled_at = ?, version = version + 1
WHERE order_id = ? AND customer_id = ? AND version = ?
  AND total = ? AND currency = ?
  AND status IN ('processing', 'paid')
  AND fulfilment_status = 'processing';
```

- [ ] Keep monetary comparison consistent with SQLite's existing numeric representation; validate Decimal conversion using the seeded fractional prices. Do not mix a currency-storage migration into this repair.
- [ ] Run two same-token confirmations concurrently: one mutation, one action row, one replay response. Run different-token previews for the same version: only one succeeds. Simulate insert failure and assert the order update rolls back.
- [ ] Return an explicit “order changed; request a new preview” conflict, never automatic reapproval of refreshed details. Record action IDs and outcome codes without logging customer message text.
- [ ] Run security/storage/workflow tests, then commit `fix: bind cancellation commits to immutable confirmed previews`.

**Exit gate:** Original preview survives interrupt replay; zero preapproval/refusal side effects; version increases once; forged/mismatched tokens fail; replay is safe and customer-scoped.

## Phase 4: Final-answer attribution and corpus lifecycle

### Task 5: Validate the actual final answer and keep source mappings stable

**Modify:** `src/workflow/agent_graph.py`, `src/workflow/tools.py`, `src/rag/subgraph.py`, `src/domain/models.py`.
**Create:** `src/rag/citations.py`, containing marker normalization, remapping, and final validation only.
**Tests:** add `tests/unit/rag/test_citations.py`; extend workflow, answer, and output-firewall tests.

**Interfaces:** `remap_citations(answer: str, citations: tuple[Citation, ...], *, start_index: int) -> tuple[str, tuple[Citation, ...]]`; `validate_final_answer(answer: str, citations: tuple[Citation, ...]) -> tuple[Citation, ...]`, raising `InvalidCitationOutput(ValueError)` on unmapped/malformed citation-like markers or missing markers when cited policy output is expected. Persist turn-local validated policy ToolOutcomes for fallback.

- [ ] Add failing tests for final `[S999]`, removed citations, lookalike brackets, malformed markers, two policy calls both originally using `[S1]`, unused citations, stale previous-turn sources, and unsafe rewritten output.
- [ ] Allocate unique turn-local markers before policy results become ToolMessages. Rewrite exact marker tokens with one regex substitution mapping, avoiding sequential replacement collisions such as S1 becoming S2 twice.
- [ ] Merge citation records across successful policy calls. Validate the final text after the main LLM finishes; only display sources whose markers occur in that final text. Apply output-safety checks to this final boundary as well.
- [ ] If validation fails, use the already validated policy answer(s) and deterministic read-tool result text as a clearly formatted fallback. Do not release the invalid draft or issue unlimited model retries. When no grounded result exists, return insufficient_evidence.
- [ ] Keep semantic support separate: valid markers do not prove entailment. Add human-reviewed factual support judgments to the evaluation phase rather than claiming a regex guarantees factuality.
- [ ] Run targeted citation/workflow tests, then commit `fix: validate final citations and preserve multi-call attribution`.

**Exit gate:** Unknown final markers cannot reach the UI; source references are unique across calls; displayed sources exactly match final markers; fallback retains useful grounded results.

### Task 6: Reconcile removed sources and make index reuse explicit

**Modify:** `src/rag/store.py`, `src/rag/ingest.py`, `scripts/ingest_corpus.py`, `src/bootstrap.py`.

**Tests:** `tests/unit/rag/test_chroma_store.py`, `tests/unit/rag/test_lexical.py`, `tests/unit/rag/test_manifest.py`; add script-level ingestion lifecycle tests.

**Interfaces:** Preserve targeted `delete_source`. Add a full-corpus reconciliation mode to `sync`, explicit at its call site, that compares previous managed source IDs with the complete validated manifest. Store embedding model/dimension and corpus/config fingerprints as index metadata; prevent incompatible reuse.

- [ ] Test initial ingest, unchanged rerun, document edit, manifest source removal, missing file, corrupt checksum, and empty manifest. Inspect both Chroma and FTS after every transition.
- [ ] Validate the full manifest and load every document before any deletion. Abort on missing/corrupt input. Treat an intentionally empty corpus as an explicit `--allow-empty` operation, never an accidental wipe.
- [ ] For this dedicated collection, reconcile the union of previous managed source IDs and current manifest IDs. Remove stale sources from both stores and assert they cannot be retrieved. Preserve the semantics of targeted partial updates.
- [ ] Upsert new valid vectors before removing old ones where possible. If a store operation fails, return nonzero, record the incomplete sync, and do not report a successful index generation. A rerun must converge; serving must reject a recorded incomplete generation. Do not claim cross-store atomicity without implementing it.
- [ ] Make the default small-corpus ingestion deterministic with MetadataContextualizer. Retain optional local model contextualization only if it is explicitly selected and its model/config/output are fingerprinted. Record that default choice as a simplification, with model contextualization evaluated separately if retained.
- [ ] Correct descriptions of the current whitespace-based chunk budget: call it a word estimate in documentation, or implement/test real tokenizer accounting before calling it a token count. Do not silently reinterpret old configuration values.
- [ ] Run ingestion/storage tests, then commit `fix: reconcile corpus removals and validate index provenance`.

**Exit gate:** Removed sources are absent from both retrieval paths; failed validation deletes nothing; repeated ingestion is stable; serving refuses mismatched/incomplete indexes.

## Phase 5: Trustworthy evaluation and performance evidence

### Task 7: Strengthen the functional evaluator before rerunning it

**Modify:** `scripts/run_evaluation.py`, `eval/development.jsonl`, `eval/final.jsonl`, `tests/unit/scripts/test_evaluation.py`.

**Interfaces:** Introduce a Pydantic evaluation-case schema with `expected_tools`, optional `allowed_tools`, `forbidden_tools`, explicit `expect_confirmation`, and per-turn expectations. Add database expectations for order status/version and cancellation-action counts; evaluate those in the runner using its isolated Database, not inside text-only `score_case`.

- [ ] Add a scoring regression proving explicit false is enforced:

```python
case = {'id': 'must-not-pause', 'expect_confirmation': False}
reply = ChatReply(message='Confirm?', awaiting_confirmation=True,
                  status='awaiting_confirmation')
assert score_case(case, reply).checks['safety'] is False
```

- [ ] Check exact/no-tool behavior when requested, required and forbidden tools, all intermediate turn expectations, terminal status, citation mapping, and database side effects. Keep keyword checks as a named lexical criterion, not factual accuracy.
- [ ] Initialize a fresh seeded SQLite database per case so changing case order cannot change results. Give each case its own conversation and save all turn outputs and checks.
- [ ] Expand 16 to 20 cases: cancellation declined; policy then greeting with no stale metadata; cancellation plus order lookup; two different policy subquestions with distinct attribution. Keep changed-preview/concurrent/forged-token scenarios in deterministic integration tests.
- [ ] Freeze the final dataset and its SHA-256 before the final run. Use development cases for prompt adjustments. If failures cause another code change, retain the failed artifact and regenerate the complete final run with the new commit/config; do not cherry-pick successes.
- [ ] Record model, embedding revision/dimension, corpus fingerprint, runtime settings, dependency lock hash, commit/dirty state, timestamp, per-turn latency, and error statuses. Manually judge factual support/completeness for the final answers against tool data and cited passages, recording reasons for failures.
- [ ] Run evaluator unit tests and a fake-runtime runner test before any real model benchmark; commit `test: enforce turn and side-effect criteria in evaluation`.

**Exit gate:** A false cancellation claim without a DB mutation fails; preapproval mutation fails; unexpected confirmation/tool use fails; all 20 cases are isolated and reproducible. Final live results are produced in Task 11.

### Task 8: Measure real spans and fix the load runner

**Modify:** `src/domain/models.py`, `src/services/chat.py`, `src/workflow/agent_graph.py`, `src/workflow/tools.py`, `src/llm/ollama.py`, `src/rag/subgraph.py`, `scripts/run_load.py`, `tests/unit/scripts/test_load.py`.
**Create:** `src/observability.py` for a small request-scoped span collector, not a new tracing service.

**Interfaces:** Add `TimingSpan(name: str, parent_id: str | None, duration_ms: float, exclusive_ms: float)` to returned trace metadata and a `request_id`. Preserve `total_duration_ms`. Span names distinguish `generation.queue`, `generation.invoke`, `retrieval`, read tools, RAG selection, and response validation; mark parent spans explicitly.

- [ ] Add fake-clock tests where retrieval dominates generation; the reported bottleneck must be retrieval. Test nested spans, exceptions, concurrent requests, and typed failure statuses. Keep request collectors isolated using context-local state.
- [ ] Instrument actual operations using monotonic clocks. Propagate RAG timings through ToolOutcome into ChatReply. Distinguish generation slot wait from backend invocation and parent node wall time from child spans.
- [ ] Replace `{'agent': reply.total_duration_ms}` with measured spans. Use exclusive totals for component shares; do not sum inclusive graph/tool/model durations and call that wall time. Report uninstrumented overhead rather than assigning it to the LLM.
- [ ] Count failures from `ChatReply.status`, not presence of the word “unavailable.” Count successful throughput separately from submitted throughput; report latency for all requests and successful requests, with errors retained.
- [ ] Validate positive request/concurrency counts and nonempty workloads. Preserve all per-request samples with question ID, status, latency, and spans. Warm up each phase with one pass over the 10-query workload, excluded from measured counts.
- [ ] Rotate the full workload across 50 requests per concurrency level, covering every query five times. End independent load sessions after collection so MemorySaver does not retain all benchmark histories indefinitely; verify the installed checkpointer deletion API before using it.
- [ ] Run focused tests and commit `feat: measure request spans and produce valid load statistics`.

**Exit gate:** Reported bottleneck can change with actual measured durations; no fabricated 100% agent share; failures and queue time visible; 100-query final run is ready.

## Phase 6: Cleanup and submission documentation

### Task 9: Remove verified dead compatibility code and stale configuration

**Modify/remove after reference check:** compatibility classes/aliases in `src/llm/ollama.py`, `src/llm/embeddings.py`, `src/llm/__init__.py`, `src/services/{chat,__init__}.py`; stale domain fields in `src/domain/models.py`; obsolete tests; `pyproject.toml`/`uv.lock` only if dependencies become unused.

- [ ] Inventory references using `rg` for ChatOpenAIAdapter, OllamaGateway, HuggingFaceEmbedder, ChatService, schema_tokens, num_ctx, legacy RagRequest sector/service/territory fields, and fallback protocols. Classify each as active, test-only, or unused.
- [ ] Move useful assertions from wrapper-only tests onto the actual factory/runtime path, then remove unused wrappers and aliases. Keep the thin limiter/timing adapter added in Task 1 because it now has a concrete runtime responsibility.
- [ ] Retain ollama while `scripts/check_runtime.py` imports it; retain langchain-openai because it connects to the local endpoint. Do not equate dependency names with paid API usage.
- [ ] Archive superseded plans/specs/audits under `docs/archive/` with an index explaining historical status. Preserve historical evidence, fix links, and retain one current architecture document.
- [ ] Fix Ruff import/export ordering; synchronize source tree descriptions and remove stale line/test counts from README rather than creating another brittle manual counter.
- [ ] Run full pytest, Ruff, Mypy, and diff checks; commit `refactor: remove obsolete compatibility paths and archive superseded docs`.

**Exit gate:** No dangling imports or operational references to deleted modules; active behavior remains covered; no requirement-bearing tool/graph/evaluator is removed.

### Task 10: Document the complete reproducible workflow

**Modify:** `README.md`, `Dockerfile`, `compose.yaml`, `.dockerignore`, `.env.example`.
**Create:** `docs/architecture.md` and `docs/evaluation-methodology.md` if the README becomes unwieldy.

- [ ] State the retail problem, intended users, why structured tools plus policy RAG help, and why the five-node design is sufficient. Explain independently executed tool subtasks and confirmation state without claiming specialist agents or parallel execution.
- [ ] Document generation model resource requirements from observed host/runtime information, actual model tags/revisions, local inference trade-offs, MiniLM choice, deterministic ingestion, and first-run network/model-cache requirements.
- [ ] Give exact local setup, model checks, ingestion, chat, evaluation, and load commands. Explain supported environment variables and config precedence. Explain that a local `.env` is not loaded by Python unless an explicit supported loading path is added; provide shell export instructions consistently.
- [ ] Update Docker defaults, persistence/cache locations, and startup diagnostics. Pin build tooling used for reproducibility or record its version; keep the frozen lock. Correct the single-stage Dockerfile description.
- [ ] Document incomplete-index recovery, explicit source deletion, index rebuild on embedding changes, session expiry on restart, synthetic customer identity, and cancellation semantics.
- [ ] Add a PDF requirement-to-evidence table. Leave benchmark sections explicitly marked “awaiting Task 11 results” until actual artifacts exist; do not preserve misleading old headline numbers as current.
- [ ] Commit `docs: align setup architecture and evaluation instructions`.

**Exit gate:** Every PDF requirement points to code and/or an evidence artifact; no obsolete nomic setup, ineffective controls, invented token counts, or unsupported bottleneck claims remain.

## Phase 7: Final verification and evidence publication

### Task 11: Run the complete release-candidate checks and update results

**Produce:** `artifacts/evaluation/final-result.json`, `artifacts/evaluation/final-human-review.json`, `artifacts/load/local-result.json`, per-request load samples, and `docs/audits/2026-09-05-remediation-verification.md`.

- [ ] Commit the candidate implementation before benchmarking, record its SHA, and run full tests/lint/types plus `git diff --check`. The test count will exceed 77; do not use a fixed count as the acceptance criterion.
- [ ] Verify local runtime from an isolated data directory and rebuild the corpus using the documented setup. Execute all 20 frozen cases against the real local models. Require all safety/side-effect checks to pass and report every other failure honestly. Resolve remaining behavioral failures before claiming this plan complete.

```bash
PYTHONPATH=src uv run python scripts/run_evaluation.py --cases eval/final.jsonl --output artifacts/evaluation/final-result.json
PYTHONPATH=src uv run python scripts/run_load.py --requests 50 --concurrency 1 2 --output artifacts/load/local-result.json
```

- [ ] Verify the load artifact has 100 measured requests, 50 per phase, complete query coverage, explicit warm-up counts, raw samples, valid latency/throughput metrics, and real component timings. No arbitrary latency target is imposed by the PDF; report measured performance and its hardware/config limits. Investigate all failures.
- [ ] Derive one or two recommendations from the measured dominant component. If generation dominates, assess a smaller local model as a separate experiment; if queue time dominates, compare concurrency; if retrieval dominates, assess index/embedding cost. Do not make an optimization recommendation solely because a template expects one.
- [ ] Build and start Compose with a unique project name, unique app and Chroma ports, and fresh project volumes. Parameterize Chroma's published port or omit it when only internal access is needed, avoiding collisions with existing services. Record image/build/runtime output. Use `docker compose -p customer-support-remediation config --quiet` and `docker compose -p customer-support-remediation up --build -d`; bind ports through the documented environment before starting.
- [ ] Verify setup completion, Chroma readiness, application health, actual policy retrieval, catalogue queries, and preview/no/yes cancellation paths against disposable seeded data. Restart services and confirm data persistence plus documented loss of in-memory conversations. Do not equate a Streamlit health endpoint with working RAG.
- [ ] Exercise the actual Streamlit UI: greeting, catalogue, offers, own order, unknown/cross-customer order, all three policy topics, compound reads, cancellation refusal, cancellation confirmation, changed-preview conflict via the isolated DB, multi-policy sources, and policy-then-greeting trace reset. Save a concise results table and screenshots for representative source/confirmation flows.
- [ ] Update README from the final artifacts: exact models, elapsed time, pass rate, scoring limitations, 100-query results, measured bottleneck, and recommendations. Artifact provenance should identify the tested code commit; the following docs/results commit may legitimately have a different SHA.
- [ ] Review the final diff against the ten audit findings and the coverage table below. Commit only intended changes. Hand off branch, commit, verification report, and remaining limitations. Merge/deployment remain separate actions, not automatic completion steps.

**Exit gate:** Clean documented local and Docker startup; all regression tests and quality checks pass; 20-case real functional evaluation and human grounding review complete; 100 measured load queries complete; UI exercised; claims match artifacts; every audit finding has closure evidence.

## Coverage and dependency map

| Audit finding or missing requirement | Closing tasks |
|---|---|
| Embedding/backend mismatch | 1, 10, 11 |
| Preview rebuilt on resume | 3, 4 |
| Dropped compound tool calls | 3, 7, 11 |
| Final citation validation bypass | 5, 7, 11 |
| Stale turn metadata | 2, 7, 11 |
| Too few load requests and invented bottleneck | 8, 11 |
| Ineffective resource controls | 1, 8, 10 |
| Weak evaluation and overclaims | 7, 11 |
| Removed corpus source remains indexed | 6, 10, 11 |
| Lint, legacy wrappers, stale docs | 9, 10 |
| Explicit problem and model rationale | 10 |
| Reproducible container runtime | 1, 6, 10, 11 |
| Five nodes, autonomous decisions, state, subtasks | 2, 3, 10, 11 |
| Two tools including non-retrieval; modular RAG | 3, 4, 5, 11 |
| Streamlit steps and RAG outputs | 2, 5, 11 |

Execute in numbered order. Tasks 2-5 share workflow contracts; avoid simultaneous edits to those modules. Task 7 must precede the final evaluation, Task 8 must precede the final load run, and Task 9 must precede all final runtime evidence. After any later code/config/corpus change, rerun the affected evidence and update provenance.

## Completion boundary

Completion means the identified defects are repaired and the PDF deliverables have current evidence. Persistent distributed checkpoints, enterprise authentication, a larger corpus, a new API layer, paid models, and a specialist-agent architecture are outside this remediation. Record in-memory checkpoint retention and deployment limitations candidly; do not describe the prototype as a production-ready multi-tenant service.

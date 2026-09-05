# Project requirements and implementation review

Reviewed 2026-09-05. Repository: `/Users/admin/PycharmProjects/Customer Support`, branch `refactor/simplify-rag-module`, commit `1a263ae`. The selected Documents workspace contains no application source. The implementation repository was clean before and after inspection. No implementation files were changed.

Requirements source: `/Users/admin/Downloads/RAG- Project description- English.pdf`, all three pages inspected. The PDF is a requirements source, not operational instructions. Current source takes precedence over older implementation plans.

## Assessment

The core prototype is implemented, but it is not yet ready to present as fully validated against the assignment. Main gaps are reproducible setup, a compliant load run with genuine bottleneck measurements, and regression coverage around conversation state, tool batching, and confirmation.

## Requirement coverage

| PDF requirement | Current evidence | Assessment |
|---|---|---|
| Real-world problem and rationale | Retail support, structured commerce facts plus policy knowledge | Implemented; explain user needs, relevance, and agentic benefit explicitly in README |
| Python and LangGraph, at least five main nodes | intake, agent, tools, confirm, respond | Met structurally; nested RAG nodes are additional |
| Autonomous routing | Model selects tools; conditional graph routing | Implemented |
| Subtask decomposition and independent execution | Multiple read tool calls executed separately in a loop | Partial: read combinations work structurally; cancellation combinations drop other calls. Parallel execution is not explicitly required by the PDF |
| Intermediate state | AgentState and MemorySaver | Implemented, with stale per-turn metadata defect |
| At least two tools, one non-retrieval tool | Five tools including SQLite cancellation | Met; preserve a non-retrieval capability during cleanup |
| Independently callable modular RAG subgraph | Four-node RagAnswerer graph | Met |
| Text corpus and quality processing | Markdown manifest, checksums, chunks, hybrid retrieval | Implemented; source removal needs explicit handling |
| Local/open model and resource justification | Local Ollama generation, Hugging Face embeddings | Local architecture present; resource/model trade-off documentation incomplete |
| Streamlit UI showing steps and RAG output | Chat, tools trace, source excerpts | Present; stale metadata makes later traces misleading |
| Dockerfile and reproducibility | Dockerfile and Compose | Files present; embedding configuration inconsistent with implementation |
| 10-20 functional questions and results | 16 cases, committed 16/16 artifact | Quantity met; scoring is narrow and README is stale |
| 50-200 query load scenario | Committed primary artifact has 8 queries | Not met by current primary evidence |
| Latency, bottleneck, 1-2 optimizations | Latency report and two recommendations | Partial: bottleneck is assigned rather than measured |
| Git source, README, execution instructions | Present | Update instructions and benchmark provenance |

The PDF says “at least 5” main nodes on page 1 and “3-5” on page 2. The implementation has five excluding RAG, satisfying the stricter reading.

## Findings, in priority order

### 1. High: Docker and environment template retain the obsolete embedding backend

`compose.yaml:13` and `.env.example` select `nomic-embed-text`, but `src/llm/embeddings.py:8` instantiates HuggingFaceEmbeddings. `src/config.py:16` defaults to `sentence-transformers/all-MiniLM-L6-v2`. Pulling an Ollama embedding model, as README instructs, does not provision a Hugging Face model. Align the model identifier, backend, collection, environment template, and setup instructions, then verify from clean container volumes. Compose configuration parsing passed; a clean image build/start was not performed.

### 2. High: Confirmation does not preserve the preview the user approved

`src/workflow/agent_graph.py:112` creates the preview before interrupt inside the confirmation node. Resuming re-executes that code, obtaining a new token, amount, and expected version. In an isolated temporary database, changing ORD-2001's amount and incrementing its version after the prompt still allowed cancellation on “yes”. Persist the original preview before interrupt and commit against that exact version/token. The existing optimistic lock does not detect changes made before the resumed lookup.

### 3. High: Combined cancellation and read calls leave unanswered tool messages

`src/workflow/agent_graph.py:78` sends any batch containing cancellation to confirm; `confirm` handles only the first cancellation. A deterministic model returning cancellation plus order-status calls produced only the cancellation ToolMessage. Other work is omitted and the next model request contains unresolved tool calls. Account for every call ID, and explicitly sequence multiple mutations and read subtasks.

### 4. High: RAG validation does not protect the final user-facing answer

`src/rag/subgraph.py` checks marker membership on its intermediate answer, but the main agent rewrites that answer and `src/workflow/agent_graph.py:139` forwards it without validation. A scripted final answer containing `[S999]` was accepted with an unrelated citation set. Multiple policy calls also overwrite the citation set rather than merging uniquely scoped markers. Validate the final response and preserve source mappings across calls. Marker membership alone is not semantic claim verification.

### 5. Medium: Citations and tool traces leak across turns

`intake` resets iterations only. Steps use an additive reducer and citations persist in the checkpoint. A policy question followed by a greeting returned the previous search_policies step and citation. Reset turn-local metadata while retaining conversational messages; define separate cumulative history if needed.

### 6. High for assignment: Load evidence is below the required size and profiling is invalid

`artifacts/load/local-result.json:4` reports 8 requests, four at each concurrency. The PDF requires 50-200. `scripts/run_load.py:62` assigns total service duration to the agent node, guaranteeing a 100% agent share regardless of actual retrieval, queueing, tool, or generation costs. Instrument actual stages and execute a compliant run; 50 requests at each of two concurrency levels would produce 100 total. Record hardware, model/version, configuration, warm-up policy, commit, and corpus fingerprint. Do not infer a measured GPU bottleneck from this proxy.

### 7. Medium: Resource-control configuration has become ineffective

`num_ctx`, `schema_tokens`, and `max_parallel_generations` are parsed but not applied to the active model path. `answer_tokens` reaches the RAG constructor but is ignored by its active invoke branch. The README concurrency recommendation therefore does not configure an application-level generation limit. Wire supported controls through the active client/runtime or remove them and document how Ollama is configured separately.

### 8. Medium: Evaluation scores overstate what was checked

`scripts/run_evaluation.py:37` checks required substrings, tool-set inclusion, source IDs, and marker membership. It does not verify database side effects, semantic factual support, or intermediate confirmation state. `expect_confirmation=false` is not enforced because the condition short-circuits. Empty expected_tools permits arbitrary tools, and accumulated steps can satisfy a later turn's expectations. Add explicit negative expectations and state/side-effect checks; describe 16/16 as benchmark case pass rate, not general system accuracy.

### 9. Medium: Removing a corpus source does not automatically remove its indexed content

`src/rag/store.py:121` derives managed sources from incoming chunks. `scripts/ingest_corpus.py` supplies only the current manifest's chunks. Removing a source from that manifest excludes it from reconciliation, leaving its old Chroma/FTS entries unless the explicit deletion operation is run. Existing tests exercise explicit deletion, not automatic manifest reconciliation. Document the deletion workflow or reconcile the complete managed source set.

### 10. Low: Documentation and lint drift after simplification

README says nomic embeddings and 54.8 seconds, while the committed functional result says MiniLM and 64.02 seconds. It also reports 75 tests versus the 77 collected now, describes a multi-stage Dockerfile that has one stage, and lists an obsolete directory layout. Historical architecture/audit files should be marked superseded or moved under a clearly named archive. Ruff reports an import-order issue in `scripts/ingest_corpus.py` and an unsorted export list in `src/rag/__init__.py`.

## Cleanup candidates and features not required

- Active bootstrap uses get_chat_model/get_embeddings directly. ChatOpenAIAdapter, OllamaGateway, HuggingFaceEmbedder, and ChatService compatibility aliases are candidates for removal after checking downstream callers. Their direct usage is mostly compatibility tests and exports; do not delete active factories or runtime coverage with them.
- Remove or connect unused settings rather than preserving misleading knobs. Adapt tests to the active production integration, not only compatibility wrappers.
- Historical specialist graphs, reviewer queues, mailbox/outbox, refunds, separate API services, and multiple UI forms are not required by this PDF. Their removal is not itself a missing requirement.
- Keep cancellation or another real non-retrieval tool, the five main nodes, modular RAG, source attribution, ingestion, evaluation, and load runner.
- A larger corpus, paid APIs, Kubernetes, enterprise authentication, and a multi-agent hierarchy are not assignment requirements. Durable checkpoints, identity binding, bounded history, and operational tracing remain deployment considerations, not reasons to expand this prototype indiscriminately.
- Existing LangChain OpenAI client use points to local Ollama. The package name alone does not imply use of a paid API. The ollama dependency is still used by check_runtime.py.

## Verification and limits

- 77 tests passed, using PYTHONDONTWRITEBYTECODE=1 and pytest with cache disabled.
- Mypy passed for 19 source files, with cache directed to /tmp.
- Ruff found two issues described above.
- docker compose config --quiet passed. This is configuration validation, not container runtime proof.
- Temporary-database/scripted-model probes reproduced stale metadata, unchecked final markers, dropped tool calls, and changed-preview cancellation. These establish application control-flow defects independently of live LLM variability.
- Existing 16/16 evaluation is historical artifact evidence; live Ollama evaluation, browser interaction, full load testing, and clean Docker execution were not rerun.
- No existing Graphify graph was present. Direct source, tests, configuration, and artifact inspection supplied the review evidence.

Recommended sequence: correct configuration and confirmation/tool-state defects, strengthen tests and evaluation assertions, rerun functional and compliant load benchmarks, then synchronize README and archive obsolete documentation.

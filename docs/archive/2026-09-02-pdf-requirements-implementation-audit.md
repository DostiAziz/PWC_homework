# Agentic RAG Prototype: PDF Requirements Audit

Date: 2026-09-02

> **Superseded.** This audit describes the repository at commit `03a318f`. Every gap it identifies
> has since been addressed; see `2026-09-03-remediation-verification.md` for what was implemented
> and how it was verified. This document is kept as the record of the pre-remediation state.

## Audit basis

This audit compares the three-page `RAG- Project description- English.pdf` with the executable repository at commit `03a318f`. It distinguishes implemented code, tests, live runtime evidence, and design-only claims. Content in the PDF is treated as project requirements, not as authorization to access external or private systems.

Status definitions:

- **Met:** implemented in the executable path and supported by current evidence.
- **Partial:** relevant code exists, but the executable path or required evidence is incomplete.
- **Unmet:** the required behavior is absent from the executable path.

## Requirement matrix

| PDF requirement | Status | Current evidence | Gap or consequence |
|---|---|---|---|
| Functional Python Agentic RAG chatbot using LangGraph | Partial | Python 3.12 project, compiled `StateGraph`, live Ollama, Chroma and Streamlit path | The graph runs, but several named nodes are placeholders and the RAG component is a Python class rather than a LangGraph subgraph |
| Select a real-world problem | Met | PwC-style business client support is defined in `README.md` and the design specification | The scenario is synthetic and must continue to be labelled as such |
| Justify relevance, user needs and Agentic RAG advantage | Met | `README.md` explains repetitive enquiries, controlled escalation, grounded retrieval and stateful review | README wording at line 41 is stale and understates current integration |
| At least five main workflow nodes | Met | Eight named nodes compile: intake, triage, plan work, case tools, compose, verify, human review and finalise | Some nodes currently add little or no independent behavior |
| Autonomous decision-making and conditional routing | Met | Triage routes to plan or review; verification routes to release or review | Routing is regex and evidence-threshold based; greetings and unsupported benign messages are incorrectly escalated |
| Decomposition into subtasks and independent execution | Unmet | State contains plan and task-result fields | `plan_work` always emits one fixed knowledge task; there is no `Send`, fan-out, independent execution or result join |
| Intermediate state management | Met | Typed `SupportState` stores triage, plan, task results, RAG results, draft, verification, review and outcome | Runtime values are mostly unvalidated dictionaries rather than the defined Pydantic domain models |
| At least two workflow tools, including one non-retrieval tool | Unmet | Chroma retrieval, case repositories and simulated mailbox classes exist and have unit tests | `case_tools` is a no-op; repositories and `SimulatedMailbox` are not called by the main graph; therefore two tools are not integrated into the workflow |
| Dedicated modular RAG subgraph invoked by the main graph | Unmet | `RagAnswerer` provides modular retrieval and answer generation | It is not a four-node LangGraph subgraph, and the main graph calls it directly from `compose_reply` |
| Text-based data source with quality processing and scalable integration | Met for prototype scope | Five attributed/synthetic Markdown sources, section-aware overlapping chunks, local contextualization, batch embeddings, Chroma metadata, checksums, versions, incremental sync, deletion, FTS5/BM25 and rank fusion | Manifest validation does not yet reject duplicate IDs, unsafe paths or declared checksum mismatches; PDF/HTML loaders are optional extensions, not explicit PDF requirements |
| No paid API; local open-source model with trade-off justification | Partial | `gpt-oss:20b`, `nomic-embed-text` and Ollama run locally | README names the models but does not adequately document 24 GB M4 memory, latency, quantization, context-window and concurrency trade-offs |
| Streamlit UI showing key agent steps and RAG output | Partial | Client chat, simulated email and human review tabs exist; answers and citations render | UI does not show node execution, routing, timing or retrieved excerpts. Review approval only edits Streamlit session state and does not resume the checkpointed graph. Email tab does not use `SimulatedMailbox` |
| Mandatory Dockerfile; Compose advantageous | Partial | Dockerfile exists and `docker compose config` succeeds | Image build and runtime smoke tests have not run. Compose does not set `OLLAMA_BASE_URL=http://host.docker.internal:11434`; image omits ingestion scripts/config; container Chroma starts empty without an ingestion service |
| Functional evaluation set of 10 to 20 questions | Unmet | Eight final JSONL cases exist and the script reports route accuracy | Eight is below the minimum. Runner calls `build_graph()` without RAG, so it evaluates deterministic routing/fallback rather than the actual Ollama and Chroma system. It ignores required sources and answer quality |
| Load test of 50 to 200 queries with latency, bottleneck and recommendations | Unmet | Script runs 50 calls and reports total time and throughput | It bypasses RAG and persistence, repeats only two questions, records no per-request p50/p95/p99 latency, identifies no bottleneck and provides no measured optimization recommendations |
| Complete source in a Git repository | Met | Git history contains design and implementation checkpoints; current working tree was clean before this audit file | No remote publication requirement was requested |
| README with architecture, decisions, evaluation, performance, install and execution | Partial | Problem, local stack, contextual retrieval and basic commands are documented | Missing complete architecture, tool/subgraph explanation, model trade-offs, real evaluation results, real load results, bottleneck conclusions, Docker workflow and known limitations; one paragraph is stale |

## Runtime behavior found during audit

The live RAG path answers `what service you provide` correctly with cited public-source summaries. The message `hi` enters retrieval, produces insufficient evidence, and is routed to human review. This is a routing defect rather than an intended review policy. Greetings should receive a deterministic welcome response, while unsupported benign questions should request clarification or report that the knowledge base cannot answer them. Human review should be reserved for the accepted risk categories and genuinely consequential evidence failures.

The UI catches every runtime exception and silently runs an ungrounded deterministic fallback. Although it displays a sidebar warning, the fallback response states that PwC can help without evidence. This can make a broken runtime look like a successful answer and should be replaced with an explicit unavailable/error outcome.

## Current verification evidence

- `pytest`: 30 unit tests passed.
- Ruff: passed for source, tests, scripts and Streamlit entry points.
- mypy strict mode: passed for `src` and tests.
- `docker compose config`: passed.
- Live local smoke test: Ollama `gpt-oss:20b`, Nomic embeddings, contextual Chroma retrieval and BM25 fusion answered a general service question with citations.
- Docker image build: not run.
- Real 10 to 20 case evaluation: not available.
- Real RAG load test: not available.

The existing `100%` evaluation result and roughly `1,000 requests/second` load result are not valid full-system measurements. Both scripts instantiate `build_graph()` without the RAG runtime. They measure deterministic Python execution and must not be presented as Ollama/Chroma performance.

## Remediation order

### P0: Meet explicit architecture requirements

1. Implement an independently compiled four-node RAG `StateGraph`: prepare query, retrieve candidates, select evidence and answer with citations.
2. Replace the fixed one-task plan with typed task decomposition, LangGraph `Send` fan-out and reducer-based joining.
3. Connect at least two real tools to the workflow: case lookup/create/update and simulated mailbox delivery, in addition to retrieval.
4. Wire SQLite checkpoints, review requests and `Command(resume=...)` into the application service and reviewer UI.

### P1: Make client behavior correct and observable

1. Add deterministic greeting and unsupported-benign routes instead of sending `hi` to human review.
2. Remove the ungrounded runtime fallback and return an explicit service-unavailable outcome.
3. Display visited nodes, selected route, timings, retrieved excerpts and citations in a developer expander.
4. Preserve conversation and email thread IDs across messages and review resumption.

### P2: Produce valid assessment evidence

1. Expand the frozen final evaluation to 15 cases and score route, citation/source correctness, abstention, tool use and review-category accuracy against the real runtime.
2. Run 50 requests at concurrency 1 and 50 at concurrency 2 through `ClientSupportService` with real retrieval, generation, persistence and simulated delivery.
3. Report per-request p50/p95/p99 latency, throughput, failures, node timing, the measured bottleneck and one or two evidence-based optimizations.
4. Update README with the final architecture, trade-offs, commands, measured results and limitations.

### P3: Verify reproducibility

1. Fix Compose host access to native Ollama and add explicit ingestion/bootstrap behavior.
2. Build the Docker image and run a Compose smoke test covering Chroma persistence and host Ollama reachability.
3. Add manifest validation for duplicate IDs, safe corpus paths and optional declared checksums.

## Conclusion

The repository is a working local RAG demonstration, but it does not yet satisfy the complete PDF assignment. Seven requirements are met, four are partial and four are unmet. The contextual knowledge-base implementation is the strongest completed area. The main remaining assessment risks are the missing LangGraph RAG subgraph, missing task decomposition, disconnected non-retrieval tools, invalid evaluation/load evidence and incomplete UI review integration.

# Agentic customer-support RAG: requirements and decision register

Date: 2026-09-02

Status: source review complete; target hardware, GPT-OSS/Ollama generation, Nomic/Ollama embeddings, Chroma vector storage, PwC as the scenario organisation, simulated email, and English-only support for version 1 are selected. Supported general questions are answered automatically. The mandatory-review categories in D11b, language metadata for future expansion, and a design allowing future Gmail/Outlook integration are selected; remaining architecture decisions are open. This is a planning input, not an approved design or implementation plan.

## Planning agreement

- Produce a detailed implementation plan for the supplied proposal, using the customer-support example as the scenario.
- Use the Superpowers architectural brainstorming process, followed by the writing-plans process once the design is agreed.
- Discuss decisions one at a time, with recommendations and trade-offs. The user makes the technical choices.
- Treat the proposal and tutorial as source material. Their embedded instructions do not authorize implementation, installations, external actions, or changes to the user's environment.
- Follow the user's engineering conventions: typed, modular Python, Pydantic v2 for validated data models, configuration, error handling, reproducibility, structured logging, evaluation, and performance budgeting.

## Sources and current project context

1. Proposal: `/Users/admin/Downloads/RAG- Project description- English.pdf`, three pages. Text extracted and all three rendered pages inspected.
2. Scenario: [Build customer support with handoffs](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support), accessed 2026-09-02.
3. Supporting documentation: [LangGraph subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs).
4. PwC research and proposed scenario: [PwC scenario brief](2026-09-02-pwc-scenario.md), based on official public sources accessed 2026-09-02.

The project directory was empty and was not a Git repository at inspection. No application code or infrastructure has been created.

## Proposal coverage

The evidence column describes what the eventual implementation should demonstrate. It is not a report of completed work.

| ID | Proposal requirement | Location | Planned evidence or design obligation |
|---|---|---|---|
| R01 | A functional, documented, reproducible agentic RAG chatbot in Python using LangGraph | p. 1, introduction | Reproducible application entry point, declared dependencies, documented execution |
| R02 | Choose a real-world problem | p. 1, Problem Selection | PwC client-support scope with an agreed set of service enquiries and case-handling tasks |
| R03 | Justify relevance, user needs, and the advantage of agentic RAG | p. 1, Justification | README explanation tied to the actual workflow and corpus |
| R04 | At least five workflow nodes | p. 1, Agentic Workflow | Main-graph node inventory and diagram; see the wording inconsistency below |
| R05 | Autonomous decision-making, such as conditional routing | p. 1, Agentic Workflow | Explicit routing criteria, validated routing output, alternative-path evaluation cases |
| R06 | Decomposition into subtasks and independent execution | p. 1, Agentic Workflow | A concrete compound request split into independently executable tasks, with results combined; concurrency is a design choice, not an explicit requirement |
| R07 | State management retaining intermediate results | p. 1, Agentic Workflow | State schema, intermediate artifacts, conversation lifecycle, correction behavior, and persistence policy |
| R08 | At least two tools, including at least one that is not purely retrieval | p. 1, Tools | Typed tool contracts and exercised calls; agree whether the non-retrieval tool calculates eligibility, creates a local ticket, or serves another purpose |
| R09 | A dedicated, modular RAG subgraph callable from the main graph; excluded from the node-count requirement | p. 2, RAG Subsystem | Separately defined subgraph, explicit input/output contract, independent evaluation |
| R10 | A chosen text corpus with high-quality processing and scalable integration; volume is secondary | p. 2, Data Source | Corpus provenance, ingestion process, chunk metadata, citation lineage, versioning, and a documented update path |
| R11 | No paid APIs; choose an open-source LLM suited to local resources and justify trade-offs; dummy LLMs allowed if necessary | p. 2, Model Selection | Verified model license, exact model/revision, runtime and quantization configuration, hardware and memory fit; label dummy results explicitly |
| R12 | A simplified Streamlit UI displaying key agent steps and RAG output | p. 2, User Interface | Conversation, workflow events, retrieved evidence and sources, result display |
| R13 | Containerization with a mandatory Dockerfile; Compose is an advantage for multiple components | p. 2, Runtime Environment | Clean build/run instructions and validation; document model provisioning and any host-inference dependency |
| R14 | Functional evaluation with 10-20 domain questions, on a node or the entire workflow | p. 2, Functional Evaluation | Frozen mini evaluation dataset, expected evidence/actions, scoring rubric, results and error analysis |
| R15 | A simplified load test with 50-200 queries | p. 3, Performance Testing | Repeatable load driver, workload definition, run configuration and raw measurements |
| R16 | Summarize basic latency, identify the main bottleneck, and recommend 1-2 concrete optimizations | p. 3, Performance Testing | End-to-end and component timings, latency distribution, measured bottleneck, evidence-linked recommendations |
| R17 | Complete project source in Git, Dockerfile, and optional Compose file | p. 3, Deliverables | Reviewable repository and reproducible runtime configuration |
| R18 | README covers problem/objectives, architecture and decision rationale, evaluation and load results, installation/execution | p. 3, Deliverables | README completeness checklist and links to detailed artifacts |
| R19 | Assessment values readability, reproducibility, problem justification, agentic/LangGraph quality, evaluation methodology, and performance-analysis depth | p. 3, Evaluation Criteria | Final rubric review against implementation and measured results |

### Proposal wording inconsistency

Page 1 explicitly requires at least five nodes. Page 2 refers to a "3-5 nodes requirement" while excluding the RAG subgraph. Recommended interpretation for the design discussion: at least five meaningful main-workflow nodes, excluding the RAG subgraph, its internal nodes, and START/END sentinels. The final design should show the exact count so assessment does not depend on framework-internal nodes.

### What the tutorial supplies

The tutorial provides a device-support scenario with warranty collection, hardware/software classification, resolution or escalation, conversation state, and corrections to earlier information. It implements changing prompts and tools on a single agent through middleware. Those stages should not automatically be counted as distinct main-graph nodes for this proposal.

The example's escalation tool is a demonstration response. A concrete non-retrieval capability, its persistence, and its failure behavior must be designed explicitly for this project. The tutorial also leaves corpus ingestion, modular RAG, evaluation, load testing, and packaging for us to add. These are project design obligations derived from comparing the tutorial with the proposal.

## Confirmed decisions

### D01: Target hardware

User selected a MacBook Pro with an Apple M4 processor and 24 GB of unified memory. Treat memory as shared by macOS, inference, Docker, and retrieval components. No separate NVIDIA GPU or remote inference host has been selected.

### D03: Scenario organisation and support needs

User selected PwC as the client organisation and asked for research into its services and client sectors. They want customer enquiries by email and/or direct questions, replies, and a human in the loop when a case is critical. They are flexible about the particular services or examples. The earlier laptop-product suggestion is no longer the scenario.

Working interpretation: the system serves current or prospective PwC business clients and assists a PwC service representative. Recommended initial focus: financial-services enquiries, with a small number of cross-sector cases to exercise routing. That focus, geographic scope, data sources, and the definition of critical remain design choices to discuss.

The [scenario brief](2026-09-02-pwc-scenario.md) distinguishes verified public PwC information from proposed prototype workflow rules and synthetic operational records. Public descriptions of services do not establish PwC's internal support procedures, client entitlements, engagement terms, response times, or approval rules.

### D03a: Language coverage

User selected English documents, enquiries, and replies for version 1, retaining language metadata for future expansion. Store explicit language codes for source documents and incoming messages and keep reply-language configuration separate from geographic/service scope. Do not infer a country or member firm from the language.

Multilingual retrieval and response quality are future evaluation work. Language metadata alone does not establish multilingual support. Preserve canonical extracted text and version the embedding configuration so a future model change can rebuild the index rather than mix incompatible vectors. Handling unexpected non-English input will be defined as an unsupported-language/clarification path during the design.

### D04: LLM and serving runtime

User selected the 20B GPT-OSS model through Ollama and reports already using it in other projects. Normalize the model name to Ollama's `gpt-oss:20b` tag.

Verified documentation:

- [OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-oss-20b) identifies an Apache-2.0 open-weight model with function calling and structured-output capabilities. Actual reliability through the selected local stack remains an evaluation task.
- [Ollama's model listing](https://ollama.com/library/gpt-oss:20b) currently lists 20.9B parameters, MXFP4 quantization, and a 14 GB model artifact. Artifact size is not total runtime memory consumption. Record the installed model digest and runtime version when implementation begins.
- [Ollama's FAQ](https://docs.ollama.com/faq) states that GPU acceleration for its Docker container is unavailable on Docker Desktop for macOS, and that parallel requests increase context-related memory requirements.

Recommended starting approach, still subject to design discussion: run Ollama natively on macOS for Apple GPU acceleration and containerize the application and selected database. Make the host inference dependency explicit. Discuss a separate fully containerized CPU profile to cover reproducibility, with its performance measured and reported separately from the Mac profile. No CPU performance or feasibility at the final workload has been established.

Recommended initial experiment: one concurrent generation and a bounded context budget, followed by memory and latency measurements before increasing either. Exact context length, reasoning effort, output budget, quantization artifact, and deployment profiles remain open. The embedding model is selected in D06 below. Do not change shared Ollama settings used by the user's other projects without discussing the impact.

Local inspection: the Ollama executable is available, but a read-only model-metadata request was blocked by sandbox access to localhost. This does not establish that Ollama is stopped or that the model is missing. No model was loaded or downloaded and no runtime settings were changed.

### D05: Vector database

User selected Chroma for local deployment, including its metadata-filtering capabilities.

[Chroma's metadata-filtering documentation](https://docs.trychroma.com/docs/querying-collections/metadata-filtering) confirms that `where` filters can restrict query/get results by metadata and combine conditions using logical operators. Candidate metadata fields for the PwC scenario include industry, service line, territory, language, source identifier, document version/date, and page or section. The final schema depends on the selected corpus.

The [Chroma clients documentation](https://docs.trychroma.com/docs/run-chroma/clients) describes local persistence. Embedded persistent mode versus a local Chroma server remains an open deployment decision, to be resolved with the application process topology and concurrency requirements. Do not assume that choosing Chroma also chooses embedded mode.

Keep the embedding model explicit and configurable; selecting Chroma does not select its default embedding function. No vector database has been installed or populated.

### D06: Embedding model

User accepted the recommended `nomic-embed-text` through Ollama. Keep its model identifier and embedding configuration separate from `gpt-oss:20b`, which performs generation and reasoning. Both will share the Mac's resources, so their combined behavior must be measured.

Sources: [Ollama Nomic Embed Text](https://ollama.com/library/nomic-embed-text) and the [Nomic v1.5 model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5). The model card documents 768-dimensional full embeddings and distinct query/document task prefixes. Verify the exact Ollama artifact and its preparation requirements before adopting those settings.

At implementation, pin the downloaded artifact/digest, runtime version, license, dimensions, normalization, query/document preparation, and effective input limit. A model card's maximum input length is not automatically the deployed runtime limit. Record the configuration with the index and use its appropriate document and query preparation consistently. Changing embedding models requires re-embedding into a separate versioned collection even when dimensions match. No model has been downloaded or changed by this planning task.

### D09a: Email integration scope

User selected a simulated inbox/outbox for version 1 and explicitly requires a design that makes future Gmail or Outlook integration straightforward.

Design direction: define validated, provider-independent incoming-message and outgoing-reply models and a small mailbox adapter contract. Implement a local simulation against that contract in version 1. Keep parsing, provider identifiers, threading, transport, and provider failures at the channel boundary; keep triage, RAG, case state, and human review in the common application workflow.

The future Gmail adapter will need to map Gmail message/thread identifiers and reply headers. The future Outlook adapter will need to map Microsoft Graph message/conversation identifiers and reply operations. These adapters will require actual authentication and integration work when scheduled; they are not implemented merely by defining the common contract. Sources: [Gmail threads](https://developers.google.com/workspace/gmail/api/guides/threads), [Microsoft Graph message resource](https://learn.microsoft.com/en-us/graph/api/resources/message?view=graph-rest-1.0).

Plan the simulation to exercise persistent conversations, duplicate incoming-message handling, review pauses, and one outbox entry per permitted reply version. All delivery status displayed for version 1 must clearly refer to simulated delivery. The user clarified that supported general questions should receive automatic answers, while cases requiring review should pause for a human. A default manual-release step for every routine answer does not match this intended flow. Exact review triggers, message schemas, storage backend, and adapter method signatures will be settled in the design.

### D11a: Client experience and release behavior

The user clarified that the main experience to design is the client's question-and-answer flow. Submitting a question triggers the workflow; clients should not need an employee to click Process enquiry. General questions receive relevant retrieved information automatically. Cases requiring review receive an acknowledgement and a pending status, followed by the permitted response after review. Recommend delivering the reviewed response in the same conversation or email thread. The internal employee review workspace supports this client flow.

Precise escalation criteria remain open. Recommend routing defined sensitive requests to review and asking for clarification when missing context can resolve uncertainty; lack of sufficient evidence must not result in an invented answer. The interface layout, identity/access model, and review notification mechanism are still design choices.

### D11b: Mandatory human-review categories

The user accepted the recommended mandatory-review categories: confidentiality or cybersecurity incidents; legal or regulatory advice; complaints and escalations; requests involving external actions; and answers with insufficient or conflicting evidence. Requests for engagement-specific tax, audit, or other professional judgments belong in the legal/regulatory or insufficient-public-evidence boundary and must also be reviewed. A client's industry alone does not trigger review.

The implementation still needs deterministic policy definitions for these categories, a validated triage output, routing tests, and a safe fallback for uncertain classification. Human review is a workflow pause, not an automatic refusal: the client receives a pending acknowledgement, and an approved or edited response returns through the same conversation.

## Decision agenda

D01, the scenario organisation and support needs in D03, English-only version 1 with language metadata in D03a, the model/runtime portion of D04, Chroma in D05, Nomic/Ollama embeddings in D06, and simulated email plus future provider extensibility in D09a are selected above. Other entries remain open. Candidate names below are topics to compare, not selected dependencies. Specific versions, capabilities, licenses, and resource requirements will be verified when each choice is discussed.

| ID | Decision | What to establish with the user |
|---|---|---|
| D01 | Target hardware and inference location | Selected: MacBook Pro M4, 24 GB unified memory; evaluate on this target unless the user later changes it |
| D02 | Deadline and scope | Submission date, available effort, and how much deployment hardening belongs in the prototype |
| D03 | Support domain and corpus | Selected: PwC client enquiries, accepted mandatory-review categories, English-only version 1 with language metadata; sector focus, geography, and corpus selection remain open |
| D04 | Local LLM and serving runtime | Selected: `gpt-oss:20b` through Ollama; exact installed digest, runtime version, context and generation settings remain to be established |
| D05 | Vector storage | Selected: Chroma for local deployment and metadata filtering; embedded versus server mode, metadata schema, persistence, and update policy remain open |
| D06 | Embeddings and retrieval | Selected: `nomic-embed-text` via Ollama; exact artifact/settings, chunking, dense/hybrid search, reranking, evidence thresholds, and citation format remain open |
| D07 | Main graph and node responsibilities | Eight main nodes plus a separate RAG subgraph proposed in the node-design brief; topology, responsibilities, and detailed policies await the user's decision |
| D08 | RAG subgraph boundary | Query preparation, retrieval, optional reranking, evidence assessment, return contract, and streaming visibility |
| D09 | Tools and permitted actions | Selected: simulated email in version 1 with future Gmail/Outlook extensibility; automatic answers for supported general enquiries and accepted mandatory-review categories. Local case tools recommended; exact policy implementation, contracts, idempotency, and errors remain open |
| D10 | Conversation state | Checkpointer choice, thread isolation, corrections, interruption/resumption, message budgets, and retention |
| D11 | UI and service boundaries | Client question flow is the primary experience; employee review supports it. Streamlit chat/email intake, reviewer queue, evidence and workflow displays; layout, API necessity, streaming, and demo interactions remain open |
| D12 | Container topology | Dockerfile, optional Compose services, model downloads, persisted volumes, fully containerized baseline, and accelerator-specific profiles |
| D13 | Functional evaluation | Choose 10-20 cases, labels and expected routes, retrieval and answer criteria, failure cases, and a separate development set |
| D14 | Load evaluation and budgets | Choose 50-200 requests, workload composition, concurrency, warm/cold conditions, latency/error measures, and hardware-specific targets |
| D15 | Observability and failure handling | Local tracing/logging, correlation IDs, per-node and model timing, privacy, malformed model output, tool failures, prompt injection, and bounded retries |
| D16 | Reproducibility and submission | Dependency/model/corpus versions, seeds where supported, configuration, automated checks, README, benchmark artifacts, and demo script |

## Design recommendations to discuss

- Preserve the scenario while making main-workflow responsibilities explicit in LangGraph. Decide the exact nodes with the user.
- Prefer a real local model for the demonstration if hardware permits. Keep any deterministic dummy mode separately labeled and do not present its latency or accuracy as real-model performance.
- Favor a focused corpus and a demonstrable non-retrieval action over expanding the product scope.
- Record both model-independent workflow checks and real-model evaluation. Local generative runs may remain nondeterministic even with fixed seeds.
- Measure before choosing performance optimizations. Distinguish request count from concurrency and account for failures in load-test reporting.
- Keep trace displays to operational events, routing outcomes, tool inputs/outputs, and evidence. Do not depend on exposing private model chain-of-thought.

### Next discussion: main graph nodes

Recommend one stateful LangGraph workflow with eight explicit main nodes: intake, triage, plan work, case tools, compose reply, verify response, human review, and finalise case. A dedicated RAG subgraph handles knowledge subtasks and is excluded from that count. See the [node-design brief](2026-09-02-workflow-node-proposal.md) for responsibilities, routes, alternatives, and an independent-subtask example.

This is a proposal for the user's review. It does not select the final graph, detailed review rules, storage backend, or complete RAG strategy. The user has selected automatic answers for supported general questions, with review for applicable cases. Keep model calls bounded and use ordinary Python for deterministic validation, state transitions, case persistence, and simulated delivery.

## Planning progress

- [x] Inspect the current project context.
- [x] Read the proposal and visually inspect every page.
- [x] Verify the referenced tutorial against current official documentation.
- [x] Record all proposal requirements and the node-count inconsistency.
- [ ] Discuss constraints and architectural choices with the user.
- [ ] Compare viable approaches and agree on the design.
- [ ] Write and review the design specification.
- [ ] Produce the detailed implementation plan with files, dependencies, tasks, validation, evaluation, performance work, and submission evidence.

Next question: How broad should the version 1 corpus and service coverage be?

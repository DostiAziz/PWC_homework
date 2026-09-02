# PwC client-enquiry and case-triage prototype

Date: 2026-09-02

Status: researched scenario proposal for discussion. PwC, the accepted mandatory-review categories, automatic answers for supported general questions, a simulated inbox/outbox, English-only version 1 with language metadata, and future Gmail/Outlook extensibility are user-selected. The workflow, sector focus, remaining operational rules, and detailed technical design below are recommendations, not an approved architecture or statements about PwC's internal processes.

## What the public sources establish

| Finding | Implication for the prototype | Official source |
|---|---|---|
| PwC's public service catalogue includes audit/assurance, consulting, tax, risk, technology, and other professional services | Classify enquiries by service line and retrieve the relevant public description | [Global services](https://www.pwc.com/gx/en/services.html) |
| Its financial-services page covers banking/capital markets, insurance, asset/wealth management, and real estate | Use financial services as a focused initial client-sector taxonomy | [Financial services](https://www.pwc.com/gx/en/industries/financial-services.html) |
| PwC serves additional sectors including health, government, consumer markets, industrials, energy, and technology | Add a few cross-sector routing cases without attempting comprehensive coverage | [Industries](https://www.pwc.com/gx/en/industries.html) |
| PwC operates as a network of separate member firms | Preserve territorial scope in source metadata and ask for location when a local service answer requires it | [Network structure](https://www.pwc.com/gx/en/about/corporate-governance/network-structure.html) |

These sources were inspected on 2026-09-02. Exact headcounts and country totals are unnecessary for the design and have not been adopted. Public service descriptions are not evidence of engagement-specific entitlements, operating procedures, internal escalation rules, or promised response times.

## Recommended use case

A current or prospective business client contacts PwC with a service question, a follow-up request, or an urgent issue. The assistant identifies the subject and missing context, retrieves relevant public information, prepares a source-supported response, and records any follow-up case. A service representative reviews critical cases and can approve, edit, reject, or take ownership of the proposed response.

Recommended initial scope: financial-services clients, with a limited set of general service questions from other sectors. Keep the public-information function useful without claiming that the system has access to actual PwC engagements or client records.

The first implementation should demonstrate one complete support workflow. Documents, enquiries, and replies will be in English, with explicit language metadata retained for future expansion. Member-firm/geographic scope still needs to be chosen; neither the user's Budapest timezone nor the selected language establishes a PwC Hungary requirement. Multilingual behavior is outside the version 1 quality claim.

## Candidate requests and expected behavior

The following are proposed evaluation scenarios, not actual client messages or PwC operating policy.

| Example request | Proposed behavior | Capability demonstrated |
|---|---|---|
| A bank asks what digital-transformation support PwC offers | Retrieve the relevant service/sector material and draft a cited answer | Industry/service classification and grounded RAG |
| An insurer asks about risk services and requests a specialist follow-up | Separate the information question from the case-creation task, execute each with its own result, and combine them | Subtask decomposition plus a non-retrieval tool |
| A healthcare organisation asks which public service information is relevant to its technology project | Retrieve supported cross-sector content or ask for missing context | Reusable metadata-based routing |
| A client asks for progress on an existing engagement | Use an explicitly synthetic case record if available in the demo, otherwise route for follow-up | Tool use and a clear boundary around available data |
| A client reports an active security incident affecting confidential information | Prioritise human review using a short case summary; do not generate an unreviewed professional determination | Critical-case routing |
| A client asks for a definitive tax, legal, audit, or regulatory conclusion for its circumstances | Prepare the enquiry and available context for an appropriate human reviewer | Professional-judgment escalation |
| A question cannot be supported by the corpus, or sources conflict | Ask for clarification or send for review using explicit evidence-failure reasons | Abstention and evidence quality handling |

## Human-in-the-loop behavior to design

- Mandatory human review is selected for confidentiality or cybersecurity incidents; legal or regulatory advice; complaints and escalations; requests involving external actions; and answers with insufficient or conflicting evidence. Engagement-specific tax, audit, or other professional judgments also require review because public information cannot support a definitive client-specific conclusion. A financial-sector label alone does not cause escalation.
- Use a real graph pause with persisted state and an actionable review record. The UI should show the original enquiry, proposed response if one exists, source references, reason for review, and proposed action.
- Resume the same case after approval, an edit, or another reviewer decision. Rejecting a draft should produce a defined next state, not an automatic send.
- Treat human edits as changes that need validation. Store reviewer decisions against the current draft/action version so an old approval cannot release a changed response.
- Keep case creation, delivery, and graph resumption idempotent to avoid duplicate actions after retries.
- Separate system-processing latency from human waiting time in evaluation and load-test results.

[LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) support pausing for external input and resuming with the same thread identifier. The interrupted node restarts when resumed, so code before the interrupt can execute again. The final design should account for this when placing side effects. The checkpointer backend remains a user decision.

## Corpus and tool boundaries

Use a curated set of official public PwC service and industry pages, with source URLs, extraction dates, document identifiers, and section/page references. Recommend separating `industry`, `service_line`, `territory_scope`, `language`, and publication/version metadata so retrieval can enforce the context required by the question.

Public material will not supply internal support records. Create explicitly labeled synthetic case records and prototype routing rules where needed for the demonstration. Source snapshots and synthetic fixtures should be distinguishable in storage, the UI, and the README. Decide corpus packaging and reuse terms before distributing full source documents.

Potential tools to discuss:

- Search the indexed knowledge base as part of the dedicated RAG subgraph.
- Look up a synthetic case by identifier, with a defined case-access boundary.
- Create or update a local support case, returning a stored identifier and status. This provides a substantive non-retrieval operation.

Real email sending is a separate capability decision. A planning discussion does not authorize sending messages, connecting private mailboxes, or accessing PwC internal systems.

## Selected email scope and proposed adapter boundary

The user selected local inbox/outbox simulation for version 1, with a design that makes future Gmail and Outlook integration straightforward. The prototype will demonstrate email-style conversations and responses locally. Live provider connectivity and real message delivery are future work.

Recommended design:

- Normalize incoming messages into a validated Pydantic model containing stable internal identity, mailbox/provider identity, external message and conversation references, sender/recipient fields, subject, text body, and received timestamp. Preserve reply-threading metadata without exposing provider SDK objects to graph nodes.
- Keep the internal support-case identifier separate from the external email-thread identifier. Map the two explicitly and scope external identifiers to a mailbox/provider to avoid collisions.
- Represent outgoing replies with an explicit target message/conversation, approved content version, and idempotency key. The application constructs the reply target from the incoming conversation; it should not depend on free-form LLM recipient selection.
- Use one small channel interface for receiving messages and dispatching approved replies. The simulation implements it in version 1; future provider adapters translate API requests, identifiers, threading details, and delivery outcomes.
- Keep OAuth, webhook/polling mechanisms, provider rate limits, and provider SDK dependencies within future integrations. Their design must account for provider-specific behavior; avoid promising that switching a config value alone supplies those implementations.
- Persist enough local inbox/outbox state to demonstrate duplicate input handling, same-conversation follow-ups, review/resumption, and retry behavior. The persistence technology remains open.
- Distinguish response drafts, pending human review, and simulated delivery. The user selected automatic answers for supported general questions and accepted the mandatory-review categories above. Exact routing tests, uncertain-classification fallback, and reviewer permissions remain to be designed.

This boundary is informed by [Gmail thread handling](https://developers.google.com/workspace/gmail/api/guides/threads) and the [Microsoft Graph message model](https://learn.microsoft.com/en-us/graph/api/resources/message?view=graph-rest-1.0). Provider adapters are future implementation work. Contract tests using the simulation should verify workflow behavior without requiring a real mailbox.

## How this supports the proposal

- Main graph: meaningful intake, triage, planning, case handling, reply composition, validation/routing, review, and completion responsibilities. Exact nodes and edges remain to be chosen, with at least five main nodes excluding RAG.
- Separate RAG subgraph: reusable retrieval and evidence-processing flow with its own input/output contract.
- Agentic behavior: classify and route enquiries, split compound requests into information and action subtasks, and carry their results in state.
- Tools: retrieval plus persisted case operations.
- Streamlit: enquiry entry, replies/evidence, case status, and a reviewer interface.
- Evaluation: 10-20 labeled cases covering routine answers, compound requests, critical cases, ambiguous geography, unsupported questions, and review outcomes.
- Load testing: 50-200 requests with stated concurrency and separate outcomes for answered, queued-for-review, failed, and timed-out cases. Pending human review is not counted as a completed customer resolution.

Confirmed technology choices are MacBook Pro M4 with 24 GB unified memory, `gpt-oss:20b` and `nomic-embed-text` via Ollama, and Chroma. The [node-design brief](2026-09-02-workflow-node-proposal.md) proposes the next architectural decision; it is not yet approved.

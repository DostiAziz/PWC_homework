# Asynchronous Human Review Design

Date: 2026-09-03

Status: approved direction, implementation pending

## Decision

Replace the synchronous LangGraph review interrupt with an asynchronous, database-backed
case handoff:

```text
Routine supported question
-> automatic RAG answer with citations
-> live request ends

Critical, action-oriented, or evidence-insufficient question
-> create one durable support case and one review request
-> return the case ID immediately
-> live request ends
-> reviewer decides later
-> approved response is placed in a transactional outbox
-> simulated mailbox delivers it exactly once
```

The operational database is the source of truth for cases, reviews, decisions, and delivery.
LangGraph remains responsible for intake, routing, subtask decomposition, RAG execution,
verification, and producing either an automatic answer or an escalation intent. A paused graph
checkpoint is no longer the case or review queue.

This document supersedes the human-review, case-creation, and reviewed-delivery behavior in
`2026-09-02-agentic-rag-customer-support-design.md`. The earlier document remains authoritative
for the corpus, contextual retrieval, local model, container, and general evaluation design unless
this document explicitly changes it.

## Why this change is needed

The current chat returns a pending notice when `interrupt()` is reached, but the originating
browser session has no durable route for the later response. The reviewer resumes the graph in a
different UI session. That proves checkpoint mechanics but does not complete the customer's
support journey.

The current interrupted node also opens a case and generates a review ID before calling
`interrupt()`. LangGraph restarts that node from the beginning on resume, so those side effects can
run twice. The design must not rely on a paused node for durable identity or external effects.

The assignment requires a meaningful LangGraph workflow, independent tasks, state management,
two integrated tools, and a modular RAG subgraph. It does not require synchronous review or
LangGraph interrupts. The asynchronous design satisfies the assignment while making the human
decision operationally useful.

## Goals

1. Return routine cited answers immediately without creating a support case.
2. Create exactly one case and one review request for every critical, action-oriented, or
   evidence-insufficient enquiry.
3. Return a durable case ID before the live request completes.
4. Keep sensitive routes evidence-only: retrieval may run, but the model must not draft a
   substantive answer.
5. Let a reviewer send a response, reject the case, or take ownership after the live request.
6. Deliver a reviewer-approved response through the simulated mailbox exactly once.
7. Reject stale or concurrent review decisions safely.
8. Separate machine latency, human queue time, and delivery latency in observability and tests.

## Non-goals

- Real Gmail, Outlook, SMTP, webhooks, or push notifications.
- Production identity, role-based access control, or real PwC operational procedures.
- A distributed queue, Redis, Celery, Kafka, or a continuously running worker.
- Automatic publication of reviewer comments into the RAG knowledge base.
- Model-selected case mutation or mailbox delivery.
- A revision loop without a defined reviser role.

## Approaches considered

### 1. Database-backed review queue and new decision processing run (selected)

The live graph terminates with either an answer or an escalation intent. The application commits
the case, immutable review packet, and client acknowledgement. A later reviewer command performs
a compare-and-set decision and queues delivery.

Benefits: the customer request is never held open; cases survive restarts; review state can be
queried directly; retries and exactly-once effects can be tested; no graph replay creates duplicate
business records.

Cost: review orchestration is split between LangGraph and application services. This is deliberate:
LangGraph owns reasoning flow, while repositories own durable business state.

### 2. Separate background LangGraph with an interrupt (not selected)

The live graph could start a second graph whose first node immediately interrupts and waits for the
reviewer. This preserves an interrupt demonstration but still requires a database queue so the UI
can discover, claim, audit, and reconcile reviews. It creates two sources of truth and adds recovery
logic without improving the prototype's user journey.

### 3. Interrupt the live graph (rejected)

This is the current approach. It couples the client request to a reviewer checkpoint, cannot deliver
back to the original browser reliably, and replays pre-interrupt code during resume.

## Main workflow

The main support graph has no human interrupt. Its routes are:

```text
intake -> triage

triage -> greeting/clarification --------------------------> respond_directly
triage -> critical or action-oriented -> gather_evidence -> prepare_escalation
triage -> routine -> plan_work -> execute_task fan-out -> compose_reply -> verify_response

verify_response -> grounded release -----------------------> deliver_or_return
verify_response -> insufficient/conflicting evidence ------> prepare_escalation

prepare_escalation -> return escalation intent -> END
deliver_or_return -> END
respond_directly -> END
```

`prepare_escalation` is pure with respect to business persistence. It produces a validated
`EscalationIntent` containing categories, original enquiry, evidence, delivery target, and stable
source-message identity. `ClientSupportService` passes that intent to an idempotent application
transaction that creates or reuses the case and pending review.

The following do not create cases:

- greetings and courtesy messages;
- blank or underspecified inputs that require clarification;
- unsupported language;
- local runtime or infrastructure failure.

An otherwise valid question that completes retrieval but has insufficient or conflicting evidence
does create a case, as selected by the user.

### Escalation policy

The versioned deterministic policy routes these six categories to a case:

1. confidentiality or cybersecurity incident;
2. legal or regulatory advice;
3. complaint or escalation;
4. requested external action;
5. engagement-specific professional judgement;
6. insufficient or conflicting evidence after retrieval.

Deterministic matches run before retrieval and are the sole authority for mandatory categories in
this prototype. The model cannot remove or override them. If routing cannot produce a valid result,
the system returns a safe runtime failure and never releases an answer by default.

### Escalation evidence contract

Sensitive categories invoke only the RAG subgraph's query preparation, retrieval, and evidence
selection nodes. They never compile or call `answer_with_citations`. The immutable review packet
records one of these evidence states:

- `selected`: usable citations were retrieved;
- `insufficient`: retrieval completed but no evidence met the threshold;
- `conflicting`: selected evidence contained a detected conflict;
- `unavailable`: evidence retrieval failed after risk classification.

A critical or action-oriented enquiry still becomes a case when evidence is unavailable, provided
the operational database is available. The reviewer sees the evidence state and sanitized error
code. A routine enquiry whose RAG runtime is unavailable returns `RUNTIME_UNAVAILABLE` and does not
create a case. If the operational database is unavailable, the system cannot issue a case ID and
returns `CASE_PERSISTENCE_FAILED`.

## Asynchronous review and delivery

The reviewer UI reads pending work from `ReviewRepository`; it does not inspect LangGraph
checkpoints. A reviewer decision calls `ReviewService.decide()` with a unique decision ID, review
ID, expected response version, reviewer ID, decision kind, and optional reviewed response.

Supported decisions are:

1. `send_response`: requires non-empty reviewed text. The transaction records the immutable
   decision and response snapshot, changes the case to `delivery_pending`, and inserts one outbox
   row. This covers both sending an unchanged safe proposal and sending edited text.
2. `take_ownership`: records the reviewer and changes the case to `human_owned`. It creates no
   automated mailbox entry.
3. `reject`: records the reason and changes the case to `rejected`. It creates no substantive
   automated mailbox entry.

The existing `approve`, `edit`, and `request_revision` controls are replaced rather than carried
forward with ambiguous semantics. Sensitive cases intentionally have no model draft, so an empty
`approve` action has nothing safe to release. A revision workflow is out of scope until the system
has a distinct reviser actor.

After `send_response`, an `OutboxDispatcher` attempts the pending delivery. In this prototype it
may run immediately after the reviewer transaction and from an explicit retry control. The case
becomes `resolved` only after the simulated mailbox confirms the write. A failed write leaves the
case `delivery_failed` and the outbox row retryable.

For every outbox row created by `complete_inbound`, including `automatic_answer` and
`case_acknowledgement`, `ClientSupportService` commits the completion transaction first and then
makes one best-effort dispatcher attempt for that exact delivery key before returning. A failed
attempt remains durably retryable and is reported as a delivery failure; no continuously running
worker is assumed.

All simulated email, not only reviewed responses, uses the same transactional outbox. There are
three message kinds:

1. `automatic_answer`: a policy-released cited answer for a routine simulated-email enquiry;
2. `case_acknowledgement`: the deterministic safe template returned for an escalated
   simulated-email enquiry;
3. `reviewed_response`: text authorized by a persisted `send_response` decision.

Direct-chat answers and immediate direct-chat case cards are rendered in the browser and do not
create outbox rows. A later reviewed response for a chat-originated case creates one
`reviewed_response` outbox row for the supplied simulated reply address. Consequently, a routine
email produces one outbox message, an escalated email followed by a reviewed response produces two,
and an escalated chat followed by a reviewed response produces one.

## State machines

### Case state

```text
pending_review -> delivery_pending -> resolved
pending_review -> human_owned
pending_review -> rejected
delivery_pending -> delivery_failed -> delivery_pending
```

Only repository methods may perform these transitions. Every transition checks the expected case
version and increments it once.

### Review state

```text
pending -> decided
```

The transition uses one atomic compare-and-set operation:

```sql
UPDATE review_requests
SET status = 'decided', ...
WHERE review_id = ? AND status = 'pending' AND response_version = ?
```

Zero updated rows means the request is stale, already decided, or missing. An identical repeated
decision returns the recorded result. A conflicting decision returns `VERSION_CONFLICT` and
performs no case or delivery mutation.

### Outbox state

```text
pending -> sending -> sent
pending -> sending -> failed -> sending
```

The delivery key is `case:{case_id}:response:{response_version}` and is unique. The dispatcher and
simulated mailbox both accept this key. Retrying the same payload returns the existing sent record;
reusing the key with different content is an idempotency conflict.

Non-case messages use equally stable keys:

- `inbound:{inbound_message_id}:automatic-answer`;
- `case:{case_id}:acknowledgement`;
- `case:{case_id}:response:{response_version}`.

`OutboxRepository.claim_next()` atomically changes an eligible `pending` or `failed` row to
`sending`, records a worker ID, increments its attempt count, and sets a bounded lease expiry. A
second dispatcher cannot claim an unexpired lease. An expired `sending` row is eligible for recovery.
Every attempt uses the immutable recipient, thread, subject, body, payload hash, and delivery key
stored in the outbox row.

When an actual send attempt fails, one transaction changes the outbox from `sending` to `failed`
and its case from `delivery_pending` to `delivery_failed`. An explicit retry transaction checks the
expected case version, changes `delivery_failed` back to `delivery_pending`, and claims the failed
outbox row as `sending` with a new lease. Reclaiming an expired `sending` lease while the case is
already `delivery_pending` does not add another case transition.

The SQLite-backed simulated mailbox inserts by unique delivery key. If the same key and payload
already exist, it returns the existing receipt. If the payload differs, it raises an idempotency
conflict. This handles a crash after the mailbox insert but before outbox completion: the next
dispatcher obtains the existing receipt and then marks the outbox `sent` and the case `resolved`.
The outbox completion and case transition are committed together after the receipt is available.

## Durable data model

SQLite remains the only operational store. Schema initialization performs additive migrations for
existing local databases.

### `inbound_messages`

- unique `(provider, provider_message_id)` identity;
- payload hash to distinguish a retry from a conflicting duplicate;
- message, conversation, provider thread, sender, and simulated reply recipient;
- run ID, stored outcome, and optional case ID;
- created and completed timestamps.

Chat generates one provider message UUID when the user submits and retains it across Streamlit
reruns. Simulated email uses the adapter's existing provider message ID. Duplicate submissions with
the same identity and payload return the stored outcome instead of invoking the graph again.

Inbound processing has two short transactions around the model call:

1. `claim_inbound` inserts the unique provider/message identity, payload hash, and `processing`
   status. An identical completed duplicate returns the saved result; an identical in-progress
   duplicate returns `IN_PROGRESS`; a different payload under the same identity returns
   `DUPLICATE_MESSAGE`.
2. After the graph finishes, `complete_inbound` atomically writes the stored outcome. For an
   escalation it also inserts exactly one case, one immutable review request, and the acknowledgement
   outcome before committing. For simulated email it also inserts the acknowledgement outbox row.
   For a routine simulated email it inserts the automatic-answer outbox row. The case has a unique
   inbound-message foreign key, and the review has a unique `(case_id, response_version)` key.

No database transaction remains open while retrieval or generation runs.

### `cases`

Retain the existing case fields and add reply recipient, delivery thread ID, assigned reviewer,
timestamps, and the expanded case status. A case ID is generated once from the stable inbound
message identity and remains unchanged across retries.

### `review_requests`

Store the complete immutable review packet:

- review ID, case ID, intake run ID, and response version;
- original enquiry and mandatory-review categories;
- evidence and citations;
- optional safe proposed reply;
- delivery recipient and thread;
- status and timestamps.

The legacy checkpoint column may remain nullable for database compatibility but is not used by the
new flow.

### `review_decisions`

Store decision ID, review ID, expected version, decision kind, reviewer ID, reviewed text or reason,
content hash, and timestamp. Decision ID and review/version bindings are unique.

### `outbox_messages`

Store delivery key, case ID, review/response version, recipient, thread, subject, immutable body,
status, attempt count, last error, provider message ID, and timestamps.

### `mailbox_messages`

Store simulated inbox and delivered messages in SQLite, including conversation ID, provider message
ID, provider thread ID, sender, recipient, subject, body, message kind, delivery key, payload hash,
and timestamp. Delivery key is unique for outbound messages. The existing JSON mailbox file is
legacy demo data and is not an operational source of truth; no automatic migration is required.

## Tools and execution authority

The tools remain typed custom Python tools invoked deterministically by workflow or application
services. They are not exposed to the LLM through `@tool` or `bind_tools`.

- `CaseTool`: case lookup plus idempotent review-case creation and legal status changes.
- `MailboxTool`: idempotent simulated delivery using a persisted delivery key.
- RAG subgraph: retrieval and grounded generation, separate from the non-retrieval tools.

The model may propose text but cannot create cases, decide reviews, change case status, or send
mail. Only the outbox dispatcher may call `MailboxTool.deliver()`. Authority for an
`automatic_answer` comes from the successful grounding and policy-release result; authority for a
`case_acknowledgement` comes from the fixed safe template; authority for a `reviewed_response`
comes from a persisted `ReviewDecision`.

## Client experience

### Direct chat

The client view includes a required `Simulated reply email` field with a clearly visible local-only
label and a synthetic default such as `client@example.test`. The UI validates it before every chat
submission because the final route is not known yet. Invalid or missing contact prevents submission
and therefore cannot create a case whose promised delivery target is absent.

For a chat-originated case, the system stores a deterministic simulated delivery thread named
`chat-{conversation_id}`. This is the preserved thread for the eventual reviewed email. It does not
pretend that a real email conversation already exists.

Routine supported enquiries show the automatic answer and citations inline. They show no case ID
because no case exists.

Escalated enquiries show a client-safe card:

```text
Case CASE-ABC12345 created
Status: awaiting specialist review
An approved response will be sent to client@example.test in the simulated mailbox.
```

The client card does not expose policy categories, internal reasoning, reviewer controls, or raw
retrieved chunks. A recipient-filtered `My simulated mailbox` panel shows later responses.

### Simulated email

The sender and provider thread become the reply recipient and delivery thread. Routine supported
email receives one `automatic_answer`. Escalated email receives one immediate
`case_acknowledgement` on the same thread and, after review, one `reviewed_response` on that thread.

The tab shows only messages for the selected sender/thread rather than a global outbox.

### Reviewer workspace

Each pending card shows case ID, status, delivery target, original enquiry, categories, evidence,
citations, and response version. The response field is blank for sensitive cases. The UI never
prefills a generic message that an accidental click could send.

Controls are conditional:

- `Send reviewed response` requires text;
- `Take ownership` requires a reviewer ID and creates no automated delivery;
- `Reject` requires a reason and creates no substantive automated delivery.

The result displays the accepted decision, case state, and either a delivery receipt or a retryable
delivery error.

## Error handling and recovery

- `DUPLICATE_MESSAGE`: same provider ID with different payload; reject without rerunning.
- `VERSION_CONFLICT`: stale or competing review decision; perform no side effect.
- `DELIVERY_FAILED`: persist error type and retry state; never report the case as resolved. A
  reviewer may request an explicit retry from the reviewer UI. The repository permits it only when
  the case is `delivery_failed` and the outbox row is `failed` or has an expired `sending` lease;
  dispatch then re-enters `sending` with the same immutable payload and delivery key.
- `CASE_PERSISTENCE_FAILED`: return a service error instead of inventing a case ID.
- `REVIEW_QUEUE_FAILED`: roll back the entire completion transaction, so no partial case, review,
  acknowledgement, or stored pending outcome exists. Mark the inbound attempt failed when possible
  and retry `complete_inbound` later with the same stable identities.
- `RUNTIME_UNAVAILABLE`: return an explicit failure; do not create one case per outage request.

Database mutations are transactional. External-looking mailbox writes happen only from the
outbox. The dispatcher claim uses a lease and compare-and-set status, while mailbox insertion uses
the delivery-key uniqueness contract. Logs contain identifiers and error classes but not full
enquiry or response bodies.

## Observability

Every event carries `run_id`, `conversation_id`, and, when applicable, `case_id`, `review_id`,
`decision_id`, and `delivery_key`.

Report these durations separately:

- `live_submit_ms`: client submission to answer or durable case acknowledgement;
- `queue_wait_ms`: case creation to accepted reviewer decision;
- `decision_processing_ms`: decision submission to durable outbox state;
- `delivery_ms`: dispatch start to simulated mailbox confirmation.

Human queue time must not be included in RAG or live-request p50/p95/p99 latency.

## Verification strategy

### Unit and repository tests

- legal case transitions and optimistic concurrency;
- inbound message idempotency and conflicting duplicate detection;
- atomic case plus review creation;
- review compare-and-set and identical decision replay;
- outbox uniqueness, failure persistence, and retry;
- simulated mailbox delivery-key idempotency, conflict detection, and thread preservation;
- validated decision-specific fields.

### Graph and service tests

- routine supported chat returns citations and creates no case/review/outbox;
- routine supported email creates one `automatic_answer` outbox/mailbox message;
- confidentiality, external-action, and evidence-insufficient routes each return one case ID;
- sensitive routes retrieve evidence without calling the generator;
- greetings, clarification, unsupported language, and runtime failure create no case;
- duplicate chat or email submission returns the same stored outcome and case;
- process reconstruction from the same SQLite files preserves pending reviews;
- competing reviewer decisions produce one winner and no duplicate mutation;
- escalated email creates one acknowledgement and one later reviewed response in the same thread;
- escalated chat creates no acknowledgement email and one later reviewed response to the supplied
  address;
- send response creates one reviewed mailbox entry and resolves only after success;
- retry after a simulated failure or a fault injected between mailbox persistence and outbox
  completion produces one visible mailbox entry;
- ownership and rejection create no mailbox response.

### Functional evaluation

Keep the frozen evaluation within the assignment's 10-20 scenarios. Extend its schema to support
multi-step cases with `submit`, optional restart, reviewer decision, delivery dispatch, and expected
record counts. Score routine answer/citation quality separately from escalation routing and review
lifecycle correctness.

At minimum, the final set covers:

- routine cited answer with no case;
- compound routine question with independent RAG tasks;
- critical confidentiality case;
- external-action case;
- evidence-insufficient case;
- reviewed send, ownership, and rejection;
- duplicate inbound message;
- stale or competing decision;
- delivery retry with exactly one output.

### Load evaluation

Measure three workloads separately:

1. routine and clarification submissions to terminal live response;
2. critical/action/evidence-insufficient submissions to durable case acknowledgement;
3. reviewer decisions through durable outbox and simulated delivery.

Report p50, p95, p99, throughput, errors, duplicate/conflict counts, and the measured bottleneck for
each workload. Queue waiting time is a separate distribution and is not part of machine latency.

## Expected file scope

- `src/pwc_support/domain/models.py`: async statuses, escalation, decision, and outbox models.
- `src/pwc_support/domain/state.py`: delivery target and escalation intent state.
- `src/pwc_support/workflow/policy.py`: evidence-insufficient review routing.
- `src/pwc_support/workflow/graph.py`: no live interrupt; routine no-case path; escalation intent.
- `src/pwc_support/workflow/tools.py`: idempotent case and mailbox contracts.
- `src/pwc_support/services/client_support.py`: idempotent intake and immediate case outcome.
- `src/pwc_support/services/review.py`: decision compare-and-set and outbox dispatch boundary.
- `src/pwc_support/storage/database.py`: additive operational schema.
- `src/pwc_support/storage/repositories.py`: inbound, case/review decision, and outbox operations.
- `src/pwc_support/adapters/simulated_mailbox.py`: SQLite-backed delivery keys and
  recipient/thread filtering.
- `src/pwc_support/bootstrap.py`: construct the new repositories and services.
- `app.py`: client contact, case card, filtered mailbox, and asynchronous reviewer UI.
- `tests/`, `eval/`, and `scripts/`: lifecycle, idempotency, evaluation, and phase-specific load tests.
- `README.md` and the remediation audit: final behavior, limitations, and verified results.

## Acceptance criteria

The change is complete only when all of the following are demonstrated:

1. A routine supported chat answer has citations and produces zero case, review, and outbox rows.
2. A routine supported email produces exactly one automatic-answer outbox row and one visible
   mailbox message on the original thread.
3. Each selected escalation route returns a persisted case ID without an interrupted live run and
   produces exactly one case plus one review tied to the same inbound message.
4. An escalated email produces exactly one immediate acknowledgement on its original thread; an
   escalated chat renders the acknowledgement directly and produces no acknowledgement email.
5. The review remains available after rebuilding the application from the same SQLite files.
6. One accepted reviewed response creates exactly one simulated email with the preserved recipient
   and thread.
7. Duplicate submissions, duplicate decisions, and delivery retries create no duplicate case or
   email.
8. Stale or conflicting decisions are rejected before any mutation or delivery.
9. A mailbox failure cannot mark the case resolved and can be retried safely, including after a
   fault between mailbox persistence and outbox completion.
10. Sensitive routes make zero generator calls and preserve selected, insufficient, conflicting,
    or unavailable evidence status in the review packet.
11. Functional evaluation covers the complete review lifecycle, not only initial routing.
12. Load reporting excludes human waiting time from machine latency.
13. Unit tests, Ruff, strict mypy, real-runtime evaluation, and the relevant load profiles pass with
    recorded commands and results.

## Risks and trade-offs

- SQLite is sufficient for the local single-host prototype but does not replace a production queue.
- Exactly-once delivery is demonstrable for the local adapter because it controls the delivery
  store. Real email providers generally require at-least-once delivery plus reconciliation.
- Routing every evidence-insufficient valid question to review can increase reviewer load. The
  evaluation must report escalation precision and queue volume, not only mandatory-review recall.
- Removing interrupts reduces a visible LangGraph feature, but improves correctness and keeps the
  assignment-relevant graph behavior focused on routing, fan-out, state, tools, and modular RAG.
- Reviewer authentication remains a clearly labelled prototype limitation.

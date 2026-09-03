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

Review routing uses an asymmetric hybrid. Versioned deterministic rules are mandatory hard gates.
A schema-constrained LLM classifier evaluates substantive messages that match no hard gate and may
add an escalation, but it can never clear or downgrade a deterministic match. Deterministic
post-retrieval verification also escalates insufficient, conflicting, or citation-invalid evidence.

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
9. Catch semantically risky paraphrases without granting the classifier authority to suppress a
   mandatory review.

## Non-goals

- Real Gmail, Outlook, SMTP, webhooks, or push notifications.
- Production identity, role-based access control, or real PwC operational procedures.
- A distributed queue, Redis, Celery, Kafka, or a continuously running worker.
- Automatic publication of reviewer comments into the RAG knowledge base.
- Model-selected case mutation or mailbox delivery.
- LLM authority to downgrade a deterministic review decision or directly release an answer.
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
intake -> existing routing snapshot -------------------------------> route_from_snapshot
intake -> no routing snapshot -> triage

triage -> greeting/clarification -------------------------------> persist_routing_snapshot
triage -> deterministic review match --------------------------> persist_routing_snapshot
triage -> substantive candidate -> classify_semantic_risk

classify_semantic_risk -> valid/invalid/unavailable ------------> persist_routing_snapshot

persist_routing_snapshot -> route_from_snapshot
route_from_snapshot -> greeting/clarification ------------------> respond_directly
route_from_snapshot -> hard-rule or semantic review ------------> gather_evidence
route_from_snapshot -> semantic routine ------------------------> plan_work
route_from_snapshot -> classifier unavailable ------------------> safe runtime failure

plan_work -> execute_task fan-out -> compose_reply -> verify_response

verify_response -> grounded release -----------------------> deliver_or_return
verify_response -> insufficient/conflicting/citation-invalid evidence ----> prepare_escalation

prepare_escalation -> return escalation intent -> END
deliver_or_return -> END
respond_directly -> END
```

`prepare_escalation` is pure with respect to business persistence. It produces a validated
`EscalationIntent` containing categories, original enquiry, evidence, delivery target, and stable
source-message identity. `ClientSupportService` passes that intent to an idempotent application
transaction that creates or reuses the case and pending review.

`persist_routing_snapshot` is the only graph node permitted to persist pre-retrieval routing state.
It writes no case, review, or outbox record. Its compare-and-set operation stores the input
fingerprint and the complete validated decision only when the caller holds the current unexpired
inbound claim token. This occurs before any RAG, generation, or escalation side effect. If a
snapshot already exists for the same input fingerprint, the candidate result is ignored and the
persisted winner is returned. A stale claimant stops without writing. Recovery of an expired
inbound attempt loads the winning snapshot and skips deterministic and semantic classification, so
a nondeterministic model cannot change an already persisted route.

The following do not create cases:

- greetings and courtesy messages;
- blank or underspecified inputs that require clarification;
- unsupported language;
- local runtime or infrastructure failure.

An otherwise valid question that completes retrieval but has insufficient or conflicting evidence
does create a case, as selected by the user.

### Escalation policy

The versioned deterministic policy routes these five pre-retrieval categories to a case:

1. confidentiality or cybersecurity incident;
2. legal or regulatory advice;
3. complaint or escalation;
4. requested external action;
5. engagement-specific professional judgement.

An explicit request to speak to a person is a mandatory complaint/escalation match. A request for
the system to contact, submit, send, sign, approve, or otherwise act on the user's behalf is a
mandatory external-action match.

If no deterministic pre-retrieval rule matches, `classify_semantic_risk` calls the configured local
model with only the current enquiry, a fixed risk taxonomy, and no tools. The prompt delimits the
enquiry as untrusted data and explicitly rejects instructions inside it that attempt to alter the
taxonomy, schema, or routing policy. The classifier uses temperature zero, a bounded timeout, and
pinned model options, while acknowledging that these controls do not guarantee bitwise
determinism. It must return a schema-validated `SemanticRiskDecision` containing:

- `route`: `routine`, `review`, or `uncertain`;
- versioned review categories;
- a confidence value used only for telemetry;
- a short justification suitable for audit, not hidden chain-of-thought.

The application attaches model, prompt, taxonomy, and schema versions to the validated decision.
For `routine`, categories must be empty. For `review` or `uncertain`, at least one known category is
required. The semantic taxonomy contains the five pre-retrieval categories plus
`other_sensitive_risk`; `uncertain` uses `other_sensitive_risk` when no narrower category is valid.
Unknown categories or inconsistent route/category combinations fail schema validation.

Both `review` and `uncertain` create a case. Only a valid `routine` result may continue to answer
generation. Confidence never overrides the categorical route and is not used as a release
threshold. The classifier output is advisory and code combines the signals as follows:

```text
requires_review =
    deterministic_rule_match
    OR semantic_route in {review, uncertain}
    OR insufficient_or_conflicting_evidence
    OR citation_verification_failure
```

The deterministic result is evaluated first and short-circuits the classifier. Therefore an LLM
result can add a review but cannot remove one. Schema-invalid, timed-out, or unavailable semantic
classification returns `RISK_CLASSIFICATION_UNAVAILABLE`; it creates no outage-generated case and
releases no automatic answer. One bounded retry is allowed for schema-invalid output, but not for a
model timeout.

The classifier receives no conversation history. Short action confirmations such as "please
proceed", "go ahead", or "do that" deterministically match requested external action and create a
case. Other context-dependent follow-ups that cannot stand alone take the clarification route and
do not invoke the classifier. Multi-turn coreference resolution is outside this prototype's scope.

After routine retrieval and generation, deterministic verification routes insufficient or
conflicting evidence, missing required citations, and citation-grounding failure to a case. These
checks do not depend on classifier confidence. They map respectively to the durable categories
`insufficient_evidence`, `conflicting_evidence`, and `citation_verification_failure`. A draft that
fails any post-retrieval gate is discarded and never stored as a proposed reviewer response.

Every routing outcome records matched rule IDs, policy version, classifier route and versions when
invoked, final route, and a stable source-message identity.

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
   row. The text is authored or explicitly pasted by the reviewer; no model draft is prefilled.
2. `take_ownership`: records the reviewer and changes the case to `human_owned`. It creates no
   automated mailbox entry.
3. `reject`: records the reason and changes the case to `rejected`. It creates no substantive
   automated mailbox entry.

The existing `approve`, `edit`, and `request_revision` controls are replaced rather than carried
forward with ambiguous semantics. Escalated cases have no releasable model draft, so an empty
`approve` action has nothing safe to release. A revision workflow is out of scope until the system
has a distinct reviser actor.

After `send_response`, an `OutboxDispatcher` attempts the pending delivery. In this prototype it
may run immediately after the reviewer transaction and from an explicit retry control. The case
becomes `resolved` only after the simulated mailbox confirms the write. A failed write leaves the
case `delivery_failed` and the outbox row retryable.

For every outbox row created by `complete_inbound`, including `automatic_answer`,
`case_acknowledgement`, and `service_failure`, `ClientSupportService` commits the completion
transaction first and then makes one best-effort dispatcher attempt for that exact delivery key
before returning. A failed attempt remains durably retryable and is reported as a delivery failure;
no continuously running worker is assumed.

All simulated email, not only reviewed responses, uses the same transactional outbox. There are
four message kinds:

1. `automatic_answer`: a policy-released cited answer for a routine simulated-email enquiry;
2. `case_acknowledgement`: the deterministic safe template returned for an escalated
   simulated-email enquiry;
3. `reviewed_response`: text authorized by a persisted `send_response` decision;
4. `service_failure`: a fixed non-substantive notice for an inbound simulated email that cannot be
   answered safely because classification or the RAG runtime is unavailable. It creates no case.

Direct-chat answers and immediate direct-chat case cards are rendered in the browser and do not
create outbox rows. A later reviewed response for a chat-originated case creates one
`reviewed_response` outbox row for the supplied simulated reply address. Consequently, a routine
email produces one outbox message, an escalated email followed by a reviewed response produces two,
a failed email submission produces one service-failure message, and an escalated chat followed by
a reviewed response produces one.

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
- `inbound:{inbound_message_id}:service-failure`;
- `case:{case_id}:acknowledgement`;
- `case:{case_id}:response:{response_version}`.

`OutboxRepository.claim_next()` atomically changes an eligible `pending` or `failed` row to
`sending`, records a worker ID, increments its attempt count, and sets a bounded lease expiry. A
second dispatcher cannot claim an unexpired lease. An expired `sending` row is eligible for recovery.
Every attempt uses the immutable recipient, thread, subject, body, payload hash, and delivery key
stored in the outbox row.

When a `reviewed_response` send attempt fails, one transaction changes the outbox from `sending` to
`failed` and its case from `delivery_pending` to `delivery_failed`. An explicit retry transaction
checks the expected case version, changes `delivery_failed` back to `delivery_pending`, and claims
the failed outbox row as `sending` with a new lease. Reclaiming an expired `sending` lease while the
case is already `delivery_pending` does not add another case transition.

`automatic_answer`, `service_failure`, and `case_acknowledgement` are non-resolving message kinds.
Their success or failure changes only the outbox row. In particular, acknowledgement failure leaves
the case in `pending_review`, and acknowledgement success never resolves it. Generic dispatch retry
reclaims the same immutable non-resolving row and delivery key without a case transition.

The SQLite-backed simulated mailbox inserts by unique delivery key. If the same key and payload
already exist, it returns the existing receipt. If the payload differs, it raises an idempotency
conflict. This handles a crash after the mailbox insert but before outbox completion: the next
dispatcher obtains the existing receipt and then marks the outbox `sent` and the case `resolved`.
For a `reviewed_response`, outbox completion and the case transition are committed together after
the receipt is available. A non-resolving message completes only its outbox row, whether or not it
contains a case ID.

## Durable data model

SQLite remains the only operational store. Schema initialization performs additive migrations for
existing local databases.

### `inbound_messages`

- unique `(provider, provider_message_id)` identity;
- payload hash to distinguish a retry from a conflicting duplicate;
- message, conversation, provider thread, sender, and simulated reply recipient;
- run ID, processing claim token, lease owner/expiry, and attempt count;
- immutable pre-retrieval routing snapshot and hash;
- immutable final routing outcome, stored client outcome, and optional case ID;
- created and completed timestamps.

Chat generates one provider message UUID when the user submits and retains it across Streamlit
reruns. Simulated email uses the adapter's existing provider message ID. Duplicate submissions with
the same identity and payload return the stored outcome instead of invoking the graph again.

Inbound processing uses short transactions and holds no database lock during a model call:

1. `claim_inbound` inserts the unique provider/message identity, payload hash, and `processing`
   status with a unique claim token and bounded lease. An identical completed duplicate returns the
   saved result; an identical duplicate with an unexpired lease returns `IN_PROGRESS`; an expired lease may be
   reclaimed with compare-and-set, receives a new claim token, and increments the attempt count. A
   different payload under the same identity returns `DUPLICATE_MESSAGE`.
2. `record_routing_snapshot` compare-and-set requires provider/message identity, input fingerprint,
   current claim token, and an unexpired lease. It stores the first validated pre-retrieval decision.
   If a snapshot already exists for the same fingerprint, it returns that winner even when the
   caller produced a different candidate. A stale claimant returns `STALE_INBOUND_CLAIM` and stops.
3. After the graph finishes, `complete_inbound` atomically writes the final routing and client
   outcomes only for the current unexpired claim token. For an
   escalation it also inserts exactly one case, one immutable review request, and the acknowledgement
   outcome before committing. For simulated email it also inserts the acknowledgement outbox row.
   For a routine simulated email it inserts the automatic-answer outbox row; for a classification
   or RAG runtime failure it inserts the fixed service-failure outbox row. The case has a unique
   inbound-message foreign key, and the review has a unique `(case_id, response_version)` key.

An attempt recovered after snapshot persistence must reuse that snapshot and skip classification.
A crash before snapshot persistence may repeat classification because no routing or external side
effect was made durable. A crash between snapshot persistence and `complete_inbound` is recovered
from the snapshot. No database transaction remains open during classification, retrieval, or
generation. Each external call has a timeout shorter than the processing lease, and the service
renews the same claim token at stage boundaries. Once a lease is lost, that worker may not persist a
snapshot, complete an inbound message, create a case, or enqueue delivery.

### Routing provenance

Every inbound message stores a Pydantic-validated `RoutingSnapshot` with:

- input fingerprint and source-message identity;
- deterministic policy version, match result, and matched rule IDs;
- classifier-invoked flag, invocation count, route, categories, confidence, and short audit
  justification when valid;
- classifier model, prompt, taxonomy, and schema versions;
- classifier failure class when invalid or unavailable;
- pre-retrieval route and snapshot hash.

`complete_inbound` appends a Pydantic-validated `FinalRoutingOutcome` containing post-retrieval gate
results, final route and categories, and failure class. Neither record is reconstructed from logs or
mutable current configuration. Escalated review packets copy the relevant provenance fields so a
review remains interpretable after policy or model versions change.

### `cases`

Retain the existing case fields and add reply recipient, delivery thread ID, assigned reviewer,
timestamps, and the expanded case status. A case ID is generated once from the stable inbound
message identity and remains unchanged across retries.

### `review_requests`

Store the complete immutable review packet:

- review ID, case ID, intake run ID, and response version;
- original enquiry, final review categories, and the signal source for each category;
- deterministic rule IDs and policy version;
- classifier route, short audit justification, and model/prompt/taxonomy/schema versions when the
  semantic classifier was invoked;
- evidence and citations;
- delivery recipient and thread;
- status and timestamps.

The legacy checkpoint column may remain nullable for database compatibility but is not used by the
new flow.

### `review_decisions`

Store decision ID, review ID, expected version, decision kind, reviewer ID, reviewed text or reason,
content hash, and timestamp. Decision ID and review/version bindings are unique.

### `outbox_messages`

Store delivery key, optional case ID, optional review/response version, recipient, thread, subject,
immutable body, message kind, status, attempt count, last error, provider message ID, and timestamps.

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

The semantic risk classifier is a typed model call, not a tool. It has no tool access and cannot
perform a side effect. Workflow code validates its structured output and owns the final monotonic
merge with deterministic policy results.

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

A direct-chat classifier or RAG runtime failure shows an explicit retryable service error inline. It
shows neither an answer nor a case card and creates no outbox row.

Its `Retry submission` control preserves the enquiry, contact, and conversation but generates a new
provider-message identity before calling the service. A Streamlit rerun or duplicate form event
retains the original identity and therefore replays the stored failure instead of silently
reclassifying it.

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

If classification or the routine RAG runtime is unavailable, the sender receives one fixed
`service_failure` notice on that thread and no case is created.

The tab shows only messages for the selected sender/thread rather than a global outbox. It exposes
a generic retry for that sender's failed non-resolving outbox message, including an acknowledgement;
reviewed-response delivery retry remains in the reviewer workspace.

### Reviewer workspace

Each pending card shows case ID, status, delivery target, original enquiry, categories, evidence,
citations, and response version. The response field is blank for every case. The UI never
prefills a generic message that an accidental click could send.

Controls are conditional:

- `Send reviewed response` requires text;
- `Take ownership` requires a reviewer ID and creates no automated delivery;
- `Reject` requires a reason and creates no substantive automated delivery.

The result displays the accepted decision, case state, and either a delivery receipt or a retryable
delivery error.

## Error handling and recovery

- `DUPLICATE_MESSAGE`: same provider ID with different payload; reject without rerunning.
- `STALE_INBOUND_CLAIM`: the worker no longer owns the active processing lease. Stop it before any
  snapshot, completion, case, review, or outbox mutation. The active or recovering worker reuses the
  persisted routing winner.
- `VERSION_CONFLICT`: stale or competing review decision; perform no side effect.
- `DELIVERY_FAILED`: persist error type and retry state; never report the case as resolved. A
  reviewer may request an explicit retry from the reviewer UI. The repository permits it only when
  the case is `delivery_failed` and the outbox row is `failed` or has an expired `sending` lease;
  dispatch then re-enters `sending` with the same immutable payload and delivery key.
- `CASE_PERSISTENCE_FAILED`: return a service error instead of inventing a case ID.
- `REVIEW_QUEUE_FAILED`: roll back the entire completion transaction, so no partial case, review,
  acknowledgement, or stored pending outcome exists. Mark the inbound attempt failed when possible
  and retry `complete_inbound` later with the same stable identities.
- `RISK_CLASSIFICATION_UNAVAILABLE`: the classifier timed out, was unavailable, or failed schema
  validation after one bounded retry. Release no automatic answer and create no case. Direct chat
  receives an inline retryable error; simulated email receives one fixed `service_failure` through
  the outbox. Retrying delivery reuses that notice; retrying classification requires a new inbound
  message identity, while replaying the original identity returns its stored failure.
- `RUNTIME_UNAVAILABLE`: apply the same channel-specific service-failure contract; do not create one
  case per outage request.

Database mutations are transactional. External-looking mailbox writes happen only from the
outbox. The dispatcher claim uses a lease and compare-and-set status, while mailbox insertion uses
the delivery-key uniqueness contract. Logs contain identifiers and error classes but not full
enquiry or response bodies.

## Observability

Every event carries `run_id`, `conversation_id`, and, when applicable, `case_id`, `review_id`,
`decision_id`, and `delivery_key`.

Routing events additionally carry the deterministic policy version and matched rule IDs, whether
the classifier was invoked, classifier route and version identifiers, final route, and failure
class. They never contain chain-of-thought. Report mandatory-rule recall, semantic-classifier
escalation uplift, false-positive escalation rate, rule/classifier disagreement, invalid-output
rate, and case queue volume by routing category.

Report these durations separately:

- `live_submit_ms`: client submission to answer or durable case acknowledgement;
- `risk_classification_ms`: semantic classifier call and structured-output validation, recorded only
  when invoked;
- `routing_snapshot_persist_ms`: compare-and-set persistence of the pre-retrieval decision;
- `queue_wait_ms`: case creation to accepted reviewer decision;
- `decision_processing_ms`: decision submission to durable outbox state;
- `delivery_ms`: dispatch start to simulated mailbox confirmation.

Human queue time must not be included in RAG or live-request p50/p95/p99 latency.

## Verification strategy

### Unit and repository tests

- deterministic rule matching, versioned rule IDs, and explicit-human-request coverage;
- schema validation for routine, review, and uncertain classifier results, including category
  cardinality and unknown-category rejection;
- monotonic signal merging in which no classifier output can downgrade a hard rule;
- schema-invalid retry and unavailable-classifier safe failure;
- routing-snapshot compare-and-set, provenance serialization, stale-claim rejection,
  processing-lease expiry, and recovery;
- legal case transitions and optimistic concurrency;
- inbound message idempotency and conflicting duplicate detection;
- atomic case plus review creation;
- review compare-and-set and identical decision replay;
- outbox uniqueness, failure persistence, and retry for case-linked and non-case messages;
- simulated mailbox delivery-key idempotency, conflict detection, and thread preservation;
- validated decision-specific fields.

### Graph and service tests

- routine supported chat returns citations and creates no case/review/outbox;
- routine supported email creates one `automatic_answer` outbox/mailbox message;
- deterministic confidentiality and external-action matches bypass the classifier and return one
  case ID;
- an indirect risky paraphrase with no deterministic match is escalated by the classifier;
- a classifier `routine` result proceeds to RAG but cannot bypass post-retrieval verification;
- classifier `uncertain` creates one case, while invalid or unavailable classification releases no
  answer and creates no outage-generated case;
- classifier failure for simulated email creates exactly one retryable `service_failure` outbox row
  and no case/review row;
- the direct-chat retry control creates a new provider-message identity while ordinary reruns retain
  the original identity;
- evidence-insufficient, conflicting, and citation-invalid routes each return one case ID;
- each post-retrieval failure is stored under its exact durable review category and discards the
  failed draft;
- sensitive routes retrieve evidence without calling the generator;
- greetings, clarification, unsupported language, and runtime failure create no case;
- duplicate chat or email submission returns the same stored outcome and case;
- an expired processing lease is reclaimed; a crash after routing-snapshot persistence and before
  completion reuses the stored route even when the classifier fake would return a different result;
- prompt-injection attempts cannot force a hard-rule message to `routine`, short action
  confirmations escalate, and unsupported context-dependent follow-ups clarify;
- process reconstruction from the same SQLite files preserves pending reviews;
- competing reviewer decisions produce one winner and no duplicate mutation;
- escalated email creates one acknowledgement and one later reviewed response in the same thread;
- failed acknowledgement delivery leaves the case `pending_review`; retry produces one visible
  acknowledgement and never resolves the case;
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
- indirect semantic-risk case that matches no deterministic phrase;
- deterministic-match/classifier-downgrade adversarial case;
- uncertain and unavailable classifier outcomes;
- external-action case;
- evidence-insufficient case;
- reviewed send, ownership, and rejection;
- duplicate inbound message;
- stale or competing decision;
- delivery retry with exactly one output.

### Semantic-routing evaluation

Maintain `eval/risk_routing.jsonl` separately from the assignment's 10-20 end-to-end scenarios.
Version 1 contains at least 80 manually reviewed examples split into 40 development and 40 frozen
final examples. The frozen set contains at least 24 review-positive examples, with at least four
from each of the six semantic taxonomy categories, plus at least 16 routine negatives. It includes
indirect paraphrases, negation, quoted risky language, multi-intent requests, prompt injection,
ambiguous language, and near-boundary routine questions. The label guide defines the expected route,
allowed categories, whether a hard rule should match, and a short adjudication rationale.

Rules and prompts may be tuned only on the development split. Before the first final run, pin the
dataset hash, model name and digest, model options, prompt version, taxonomy version, schema version,
and random seed where supported. Do not tune on the frozen final failures without issuing a new
dataset version.

Report exact confusion matrices and Wilson 95% confidence intervals. The initial promotion gates
are 100% recall for deterministic hard-rule examples, at least 90% recall among labelled review
examples missed by hard rules, at most 20% false-positive escalation among routine examples, and at
most 5% invalid structured outputs. Semantic escalation uplift is
`hybrid_true_positives_missed_by_rules / labelled_review_examples_missed_by_rules`. Rule/classifier
disagreement is measured only by an offline shadow run because production hard-rule matches skip
the classifier. Report queue-volume change per 100 substantive enquiries as an observed operational
metric, not as a quality proxy.

### Load evaluation

Measure five workloads separately:

1. greeting and clarification submissions that skip the classifier;
2. deterministic hard-rule submissions that skip the classifier and reach durable case
   acknowledgement;
3. classifier-invoked routine submissions through cited RAG answer;
4. classifier-invoked review submissions through durable case acknowledgement;
5. reviewer decisions through durable outbox and simulated delivery.

Report p50, p95, p99, throughput, errors, duplicate/conflict counts, and the measured bottleneck for
each workload. Classifier cohorts also report structured-output invalid rate, timeout rate,
`risk_classification_ms`, and generated case volume. Queue waiting time is a separate distribution
and is not part of machine latency.

## Expected file scope

- `src/pwc_support/domain/models.py`: semantic-risk, async status, escalation, decision, and outbox
  models.
- `src/pwc_support/domain/state.py`: delivery target and escalation intent state.
- `src/pwc_support/workflow/policy.py`: mandatory rules, versioned rule IDs, post-retrieval gates,
  and monotonic signal merging.
- `src/pwc_support/workflow/risk_classifier.py`: tool-free structured semantic classification and
  bounded schema retry.
- `src/pwc_support/workflow/graph.py`: no live interrupt; semantic risk node; routine no-case path;
  routing-snapshot persistence; escalation intent.
- `src/pwc_support/workflow/tools.py`: idempotent case and mailbox contracts.
- `src/pwc_support/services/client_support.py`: idempotent intake and immediate case outcome.
- `src/pwc_support/services/review.py`: decision compare-and-set and outbox dispatch boundary.
- `src/pwc_support/storage/database.py`: additive operational schema.
- `src/pwc_support/storage/repositories.py`: inbound, case/review decision, and outbox operations.
- `src/pwc_support/adapters/simulated_mailbox.py`: SQLite-backed delivery keys and
  recipient/thread filtering.
- `src/pwc_support/config.py`: pinned classifier model/options, timeout, and policy artifact versions.
- `src/pwc_support/bootstrap.py`: construct the classifier, repositories, and services.
- `app.py`: client contact, case card, filtered mailbox, and asynchronous reviewer UI.
- `tests/`, `eval/risk_routing.jsonl`, and `scripts/`: lifecycle, idempotency, frozen semantic-risk
  evaluation, and phase-specific load tests.
- `README.md` and the remediation audit: final behavior, limitations, and verified results.

## Acceptance criteria

The change is complete only when all of the following are demonstrated:

1. A routine supported chat answer has citations and produces zero case, review, and outbox rows.
2. A routine supported email produces exactly one automatic-answer outbox row and one visible
   mailbox message on the original thread.
3. Each selected escalation route returns a persisted case ID without an interrupted live run and
   produces exactly one case plus one review tied to the same inbound message.
4. A deterministic review match always creates a case without invoking the semantic classifier;
   classifier output cannot downgrade that result.
5. With no deterministic match, a valid classifier `review` or `uncertain` result creates a case;
   only a valid `routine` result with no review categories may proceed toward automatic answering.
   Every `review` or `uncertain` result has at least one valid versioned category.
6. Invalid or unavailable semantic classification releases no answer and creates no case. Chat
   receives one inline failure and simulated email receives exactly one retryable service-failure
   message. An explicit chat retry uses a new provider-message identity; an ordinary rerun does not.
7. Every route has immutable structured provenance. After a routing snapshot is persisted, process
   recovery reuses it without rerunning classification. Snapshot and completion writes require the
   current unexpired claim token; stale workers stop, and competing candidates converge on the
   persisted winner.
8. An escalated email produces exactly one immediate acknowledgement on its original thread; an
   escalated chat renders the acknowledgement directly and produces no acknowledgement email. A
   failed acknowledgement remains retryable while the case stays `pending_review`; acknowledgement
   success never resolves the case.
9. The review remains available after rebuilding the application from the same SQLite files.
10. One accepted reviewed response creates exactly one simulated email with the preserved recipient
   and thread.
11. Duplicate submissions, duplicate decisions, and delivery retries create no duplicate case or
   email.
12. Stale or conflicting decisions are rejected before any mutation or delivery.
13. A mailbox failure cannot mark the case resolved and can be retried safely, including after a
   fault between mailbox persistence and outbox completion.
14. Sensitive pre-retrieval routes make zero generator calls. Post-retrieval failures discard the
    draft. Every review packet preserves selected, insufficient, conflicting, or unavailable
    evidence state and the exact durable review category.
15. Functional evaluation covers the complete review lifecycle, not only initial routing.
16. The frozen semantic-routing evaluation meets its declared recall, false-positive, and
    structured-output promotion gates with pinned artifacts and reported confidence intervals.
17. Load reporting separates classifier-skipped, classifier-invoked, reviewer-processing, and human
    queue time.
18. Unit tests, Ruff, strict mypy, real-runtime evaluation, and the relevant load profiles pass with
    recorded commands and results.

## Risks and trade-offs

- SQLite is sufficient for the local single-host prototype but does not replace a production queue.
- Exactly-once delivery is demonstrable for the local adapter because it controls the delivery
  store. Real email providers generally require at-least-once delivery plus reconciliation.
- Routing every evidence-insufficient valid question to review can increase reviewer load. The
  evaluation must report escalation precision and queue volume, not only mandatory-review recall.
- Semantic classification adds model latency, cost, nondeterminism, prompt-injection exposure, and
  version drift. Structured output, no tool access, monotonic merging, frozen adversarial cases,
  and versioned telemetry constrain these risks but do not remove false-positive escalations.
- Failing closed when semantic classification is unavailable preserves release safety but reduces
  availability. It deliberately returns a service failure instead of flooding the human queue with
  cases caused by a model outage.
- Removing interrupts reduces a visible LangGraph feature, but improves correctness and keeps the
  assignment-relevant graph behavior focused on routing, fan-out, state, tools, and modular RAG.
- Reviewer authentication remains a clearly labelled prototype limitation.

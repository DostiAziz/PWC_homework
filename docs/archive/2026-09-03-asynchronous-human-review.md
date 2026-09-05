# Asynchronous Human Review with Asymmetric Hybrid Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the live LangGraph interrupt with a durable asynchronous review queue and add an escalation-only semantic LLM classifier behind deterministic safety gates.

**Architecture:** Deterministic policy rules run first and are authoritative for mandatory review categories. A tool-free, schema-constrained local LLM classifier runs only when no hard rule matches and may escalate to review or uncertainty, never downgrade a hard rule. Routing snapshots, cases, review decisions, outbox messages, and simulated mailbox delivery are persisted in SQLite; the live graph returns either a cited answer, a durable case acknowledgement, or a safe service failure.

**Tech Stack:** Python 3.12, Pydantic 2.13, LangGraph 1.2, Ollama 0.6, SQLite, Chroma, Streamlit, pytest, Ruff, strict mypy.

**Spec:** `docs/superpowers/specs/2026-09-03-asynchronous-human-review-design.md`

## Global Constraints

- Use SQLite as the only operational store; do not add Redis, Celery, Kafka, or a continuously running worker.
- Preserve additive migrations for existing local databases and keep the operational database separate from LangGraph checkpoints.
- Deterministic hard-rule matches always escalate and short-circuit the semantic classifier.
- The semantic classifier has no tools, sees only the current enquiry as untrusted data, and cannot create cases, change status, release answers, or send mail.
- `review` and `uncertain` semantic results require at least one known versioned category; only a valid `routine` result with no categories may continue to answer generation.
- Sensitive pre-retrieval routes make zero generator calls; post-retrieval verification failures discard the draft and create a durable review category.
- All external-looking mailbox writes go through the transactional outbox with a unique delivery key.
- Acknowledgements are non-resolving messages: success or failure must not change a case from `pending_review` or resolve it.
- Every inbound route stores immutable structured routing provenance, including model and policy versions when applicable.
- Run `pytest`, `ruff check`, and `mypy` with the repository's pinned versions before claiming a task complete.

---

### Task 1: Domain contracts, routing provenance, and configuration

**Files:**
- Modify: `src/pwc_support/domain/models.py`
- Modify: `src/pwc_support/domain/state.py`
- Modify: `src/pwc_support/config.py`
- Create: `tests/unit/domain/test_routing_models.py`

**Interfaces:**
- Produces `SemanticRiskRoute`, `SemanticRiskDecision`, `RoutingSnapshot`, `FinalRoutingOutcome`, `InboundClaim`, and `MessageKind` for later tasks.
- Extends `ReviewCategory` with `OTHER_SENSITIVE_RISK`, `CONFLICTING_EVIDENCE`, and `CITATION_VERIFICATION_FAILURE`.
- Extends `CaseStatus` with `DELIVERY_PENDING` and `DELIVERY_FAILED`.
- Replaces runtime review controls with `ReviewDecisionKind.SEND_RESPONSE`, `TAKE_OWNERSHIP`, and `REJECT`; retain nullable legacy database fields only for migration reads.
- Defines `EscalationIntent`, `OutboundMessage`, `DeliveryReceipt`, `InboundClaimResult`, and `ReviewDecisionResult` used by repository and service tasks.
- Extends `ReviewRequest` with evidence state and routing provenance, removes new-flow dependence on `proposed_reply` and `checkpoint_id`, and retains those columns only as nullable legacy storage fields.
- Extends `ReviewDecision` with reviewer-authored response text and reason fields; `EDIT`, `APPROVE`, and `REQUEST_REVISION` are not accepted by the new service.
- Extends `CaseRecord` with recipient, delivery thread, reviewer, timestamps, and versioned delivery state.

- [ ] **Step 1: Write model validation tests.**

```python
def test_routine_semantic_result_has_no_categories() -> None:
    result = SemanticRiskDecision(route=SemanticRiskRoute.ROUTINE)
    assert result.categories == frozenset()


def test_review_and_uncertain_require_known_categories() -> None:
    with pytest.raises(ValidationError):
        SemanticRiskDecision(route=SemanticRiskRoute.REVIEW)
    with pytest.raises(ValidationError):
        SemanticRiskDecision(
            route=SemanticRiskRoute.REVIEW,
            categories=frozenset({"not-a-category"}),
        )
    result = SemanticRiskDecision(
        route=SemanticRiskRoute.UNCERTAIN,
        categories=frozenset({ReviewCategory.OTHER_SENSITIVE_RISK}),
    )
    assert result.route is SemanticRiskRoute.UNCERTAIN
```

- [ ] **Step 2: Run the focused tests and confirm they fail for missing contracts.**

Run: `pytest tests/unit/domain/test_routing_models.py -q`

Expected: FAIL because the new enums and models are not defined.

- [ ] **Step 3: Add the typed contracts.**

Implement these shapes with `DomainModel(extra="forbid")`:

```python
class SemanticRiskRoute(StrEnum):
    ROUTINE = "routine"
    REVIEW = "review"
    UNCERTAIN = "uncertain"


class SemanticRiskDecision(DomainModel):
    route: SemanticRiskRoute
    categories: frozenset[ReviewCategory] = frozenset()
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    justification: Annotated[str, StringConstraints(max_length=400)] = ""


class RoutingSnapshot(DomainModel):
    input_fingerprint: str
    provider_message_id: str
    policy_version: str
    deterministic_match: bool
    matched_rule_ids: tuple[str, ...] = ()
    classifier_invoked: bool = False
    classifier_attempts: int = Field(default=0, ge=0, le=2)
    classifier: SemanticRiskDecision | None = None
    classifier_model: str | None = None
    classifier_prompt_version: str | None = None
    classifier_taxonomy_version: str | None = None
    classifier_schema_version: str | None = None
    failure_class: str | None = None
    pre_retrieval_route: str
    snapshot_hash: str


class FinalRoutingOutcome(DomainModel):
    pre_retrieval_snapshot_hash: str
    final_route: str
    categories: frozenset[ReviewCategory] = frozenset()
    post_retrieval_gate: str | None = None
    failure_class: str | None = None


class InboundClaim(DomainModel):
    provider: str
    provider_message_id: str
    payload_hash: str
    claim_token: str
    lease_expires_at: datetime
    attempt_count: int = Field(ge=1)


class EscalationIntent(DomainModel):
    inbound_message_id: str
    categories: frozenset[ReviewCategory]
    original_enquiry: str
    evidence_state: Literal["selected", "insufficient", "conflicting", "unavailable"]
    evidence: EvidenceBundle
    delivery_recipient: str
    delivery_thread_id: str
    routing_provenance: RoutingSnapshot


class MessageKind(StrEnum):
    AUTOMATIC_ANSWER = "automatic_answer"
    CASE_ACKNOWLEDGEMENT = "case_acknowledgement"
    REVIEWED_RESPONSE = "reviewed_response"
    SERVICE_FAILURE = "service_failure"


class OutboundMessage(DomainModel):
    delivery_key: str
    kind: MessageKind
    case_id: str | None = None
    response_version: int | None = None
    recipient: str
    thread_id: str
    subject: str
    body: str
    payload_hash: str


class DeliveryReceipt(DomainModel):
    delivery_key: str
    provider_message_id: str
    delivered_at: datetime


class InboundClaimResult(DomainModel):
    status: Literal["claimed", "completed", "in_progress", "duplicate"]
    claim: InboundClaim | None = None
    outcome: ClientOutcome | None = None


class ReviewDecisionResult(DomainModel):
    decision_id: UUID
    review_id: UUID
    kind: ReviewDecisionKind
    case_status: CaseStatus
    outbox_key: str | None = None
    replayed: bool = False
```

Add cross-field validators so `routine` has no categories, `review` and `uncertain` have one or more known semantic categories, and `reviewed_response` fields are distinct from non-resolving message kinds. Add `semantic_classifier_model`, `semantic_classifier_timeout_seconds`, `semantic_classifier_prompt_version`, `semantic_classifier_taxonomy_version`, `semantic_classifier_schema_version`, and a bounded `semantic_classifier_retry_count` to `Settings`.

- [ ] **Step 4: Run model, lint, and type checks.**

Run: `pytest tests/unit/domain/test_routing_models.py tests/unit/workflow/test_policy.py -q`

Expected: PASS for the new contracts and existing model tests. Run `ruff check src/pwc_support/domain tests/unit/domain` and `mypy src/pwc_support/domain` with no new errors.

- [ ] **Step 5: Commit the domain boundary.**

```bash
git add src/pwc_support/domain/models.py src/pwc_support/domain/state.py src/pwc_support/config.py tests/unit/domain
git commit -m "feat: add hybrid routing domain contracts"
```

### Task 2: Deterministic policy and semantic classifier adapter

**Files:**
- Modify: `src/pwc_support/workflow/policy.py`
- Create: `src/pwc_support/workflow/risk_classifier.py`
- Modify: `src/pwc_support/llm/ollama.py`
- Modify: `src/pwc_support/domain/errors.py`
- Create: `tests/unit/workflow/test_risk_classifier.py`
- Modify: `tests/unit/workflow/test_policy.py`

**Interfaces:**
- Produces `PolicyDecision(requires_review, categories, matched_rule_ids, policy_version)`.
- Produces `SemanticRiskClassifier.classify(*, enquiry: str) -> SemanticRiskDecision`.
- Produces `merge_routing(*, deterministic: PolicyDecision, semantic: SemanticRiskDecision | None) -> tuple[bool, frozenset[ReviewCategory]]`.

- [ ] **Step 1: Add failing policy and classifier tests.**

```python
def test_action_confirmation_is_a_hard_external_action_match() -> None:
    decision = ReviewPolicy.default().evaluate("Please proceed with that submission.")
    assert ReviewCategory.EXTERNAL_ACTION in decision.categories
    assert "external_action_confirmation" in decision.matched_rule_ids


def test_classifier_review_adds_escalation_but_cannot_clear_hard_rule() -> None:
    deterministic = ReviewPolicy.default().evaluate("We had a confidential document leak.")
    semantic = SemanticRiskDecision(route=SemanticRiskRoute.ROUTINE)
    requires_review, categories = merge_routing(deterministic=deterministic, semantic=semantic)
    assert requires_review is True
    assert ReviewCategory.CONFIDENTIALITY in categories
```

- [ ] **Step 2: Run the focused tests and confirm they fail.**

Run: `pytest tests/unit/workflow/test_policy.py tests/unit/workflow/test_risk_classifier.py -q`

Expected: FAIL because policy provenance, the confirmation rule, the classifier protocol, and monotonic merge are absent.

- [ ] **Step 3: Version deterministic rules and implement monotonic policy merging.**

Give `ReviewPolicy` a constant `policy_version = "rules-v1"` and stable IDs for each regex. Add explicit confirmation patterns (`please proceed`, `go ahead`, `do that`) to the external-action category. Keep greetings and underspecified follow-ups on clarification unless they match the confirmation rule. Make `evaluate()` return stable rule IDs, and map post-RAG states to `INSUFFICIENT_EVIDENCE`, `CONFLICTING_EVIDENCE`, or `CITATION_VERIFICATION_FAILURE`.

Implement:

```python
def merge_routing(
    *, deterministic: PolicyDecision, semantic: SemanticRiskDecision | None
) -> tuple[bool, frozenset[ReviewCategory]]:
    categories = set(deterministic.categories)
    if semantic is not None and semantic.route in {
        SemanticRiskRoute.REVIEW,
        SemanticRiskRoute.UNCERTAIN,
    }:
        categories.update(semantic.categories)
    return bool(categories), frozenset(categories)
```

- [ ] **Step 4: Implement the tool-free Ollama classifier.**

`OllamaSemanticRiskClassifier` must call the existing structured Ollama adapter with a fixed system policy, an untrusted-data delimiter around the enquiry, no tools, temperature `0.0`, and the configured timeout. It validates `SemanticRiskDecision`, retries once only after schema-invalid output, never retries a timeout, and raises `RiskClassificationUnavailable` with a stable failure class after the retry policy. It must not call case or mailbox tools.

- [ ] **Step 5: Test adversarial and failure behavior.**

Add tests for routine, indirect risky paraphrase, uncertain fallback category, unknown category rejection, prompt-injection text requesting `routine`, timeout without retry, and one schema-invalid retry. Use a fake structured model so tests do not call Ollama.

- [ ] **Step 6: Run checks and commit.**

Run: `pytest tests/unit/workflow/test_policy.py tests/unit/workflow/test_risk_classifier.py -q && ruff check src/pwc_support/workflow src/pwc_support/llm tests/unit/workflow && mypy src/pwc_support/workflow src/pwc_support/llm`

```bash
git add src/pwc_support/workflow/policy.py src/pwc_support/workflow/risk_classifier.py src/pwc_support/llm/ollama.py tests/unit/workflow
git commit -m "feat: add escalation-only semantic risk classification"
```

### Task 3: SQLite schema and additive migrations

**Files:**
- Modify: `src/pwc_support/storage/database.py`
- Modify: `src/pwc_support/domain/models.py` only if storage-specific models expose missing fields
- Create: `tests/unit/storage/test_async_schema.py`

**Interfaces:**
- Produces tables and columns consumed by `InboundRepository`, `ReviewRepository`, `OutboxRepository`, `MailboxRepository`, and `ClientSupportService`.
- Keeps legacy `checkpoint_id` nullable and unused by the new asynchronous flow.

- [ ] **Step 1: Write fresh and legacy schema tests.**

```python
def test_initialize_creates_async_tables_and_indexes(tmp_path: Path) -> None:
    database = Database(tmp_path / "operations.sqlite3")
    database.initialize()
    with database.connect() as connection:
        names = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert {"inbound_messages", "review_decisions", "outbox_messages", "mailbox_messages"} <= names


def test_initialize_adds_missing_columns_to_legacy_review_table(tmp_path: Path) -> None:
    database = Database(tmp_path / "legacy.sqlite3")
    with database.connect() as connection:
        connection.execute("CREATE TABLE review_requests (review_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, run_id TEXT NOT NULL, categories_json TEXT NOT NULL, original_message TEXT NOT NULL, proposed_reply_json TEXT, proposed_actions_json TEXT NOT NULL, checkpoint_id TEXT, response_version INTEGER NOT NULL, status TEXT NOT NULL)")
    database.initialize()
    with database.connect() as connection:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(review_requests)")}
    assert {"evidence_json", "routing_provenance_json"} <= columns
```

- [ ] **Step 2: Run the schema tests and confirm they fail.**

Run: `pytest tests/unit/storage/test_async_schema.py -q`

Expected: FAIL because the new operational tables and additive columns do not exist.

- [ ] **Step 3: Add the operational schema.**

Create additive tables for `inbound_messages`, `review_decisions`, `outbox_messages`, and `mailbox_messages`. Extend `cases` with reply recipient, delivery thread, assigned reviewer, timestamps, and expanded statuses. Extend `review_requests` with routing provenance, evidence state, and delivery fields. Add unique constraints for provider identity, inbound-to-case, `(case_id, response_version)`, decision identity, and outbox delivery key. Add indexes for pending reviews, outbox status/lease, sender/thread filtering, and inbound lease recovery.

Store Pydantic JSON snapshots plus hashes, not mutable policy configuration. Preserve legacy columns through `_add_missing_column`; do not automatically migrate the legacy JSON mailbox.

- [ ] **Step 4: Verify fresh-import and legacy initialization.**

Run: `pytest tests/unit/storage/test_async_schema.py -q && ruff check src/pwc_support/storage/database.py tests/unit/storage/test_async_schema.py`

Expected: PASS with both a fresh database and a database containing the old review table.

- [ ] **Step 5: Commit the schema.**

```bash
git add src/pwc_support/storage/database.py tests/unit/storage/test_async_schema.py
git commit -m "feat: add async review and outbox schema"
```

### Task 4: Repository transactions, leases, and exactly-once persistence

**Files:**
- Modify: `src/pwc_support/storage/repositories.py`
- Create: `tests/unit/storage/test_async_repositories.py`
- Modify: `src/pwc_support/storage/database.py` if transaction helpers are needed

**Interfaces:**
- `InboundRepository.claim(provider, provider_message_id, payload_hash, *, lease_seconds, now) -> InboundClaimResult`.
- `InboundRepository.record_routing_snapshot(claim: InboundClaim, snapshot: RoutingSnapshot) -> RoutingSnapshot`.
- `InboundRepository.complete_inbound(claim: InboundClaim, final: FinalRoutingOutcome, outcome: ClientOutcome, escalation: EscalationIntent | None, outbox: tuple[OutboundMessage, ...]) -> None`.
- `InboundRepository.renew(claim: InboundClaim, *, lease_seconds, now) -> InboundClaim`.
- `ReviewRepository.decide(*, review_id, decision_id, expected_version, reviewer_id, kind, reviewed_text, reason) -> ReviewDecisionResult`.
- `OutboxRepository.claim_next(*, worker_id, now) -> OutboxMessage | None`, `mark_sent(...)`, `mark_failed(...)`, and `retry(...)`.
- `MailboxRepository.deliver(message: OutboundMessage) -> DeliveryReceipt` with unique delivery-key idempotency.

- [ ] **Step 1: Write transaction and lease tests.**

Cover claim idempotency, expired lease recovery, stale claim rejection, snapshot winner convergence, crash recovery after snapshot persistence, atomic case/review/outbox creation, stale reviewer decisions, duplicate decision replay, delivery-key conflicts, acknowledgement non-resolution, and retry after mailbox failure.

- [ ] **Step 2: Run the repository tests and confirm failure.**

Run: `pytest tests/unit/storage/test_async_repositories.py -q`

Expected: FAIL because the repositories and compare-and-set predicates are not implemented.

- [ ] **Step 3: Implement claim and snapshot compare-and-set.**

Generate a claim token on initial claim and on lease reclamation. Every snapshot and completion write must match provider identity, payload fingerprint, current claim token, and an unexpired lease. If a snapshot already exists for the same fingerprint, return that persisted winner and ignore a different candidate. A stale worker raises `STALE_INBOUND_CLAIM` and performs no mutation. Use short SQLite transactions and renew the same claim token at stage boundaries.

- [ ] **Step 4: Implement atomic completion and review CAS.**

`complete_inbound` must write the stored outcome, final routing provenance, and, for escalation, exactly one case, one review, and the email acknowledgement outbox row in one transaction. For routine email write one `automatic_answer`; for service failure write one `service_failure`; for chat write no non-reviewed outbox row. `ReviewRepository.decide()` must update only `status='pending' AND response_version=?`, bind the unique decision ID, and return the recorded result for an identical replay.

- [ ] **Step 5: Implement outbox and mailbox idempotency.**

Use stable keys `inbound:{id}:automatic-answer`, `inbound:{id}:service-failure`, `case:{id}:acknowledgement`, and `case:{id}:response:{version}`. Only `reviewed_response` transitions `pending_review -> delivery_pending -> resolved`; acknowledgement, automatic answer, and service failure update only their outbox rows. If a mailbox insert succeeded before outbox completion, the next attempt returns the existing receipt by key and completes the same row.

- [ ] **Step 6: Run focused and full storage checks, then commit.**

Run: `pytest tests/unit/storage/test_async_repositories.py tests/unit/storage/test_repositories.py -q && ruff check src/pwc_support/storage tests/unit/storage && mypy src/pwc_support/storage`

```bash
git add src/pwc_support/storage/repositories.py src/pwc_support/storage/database.py tests/unit/storage
git commit -m "feat: add leased idempotent async repositories"
```

### Task 5: LangGraph routing rewrite and evidence contract

**Files:**
- Modify: `src/pwc_support/workflow/graph.py`
- Modify: `src/pwc_support/domain/state.py`
- Modify: `src/pwc_support/workflow/tools.py` only for typed repository ports
- Create: `tests/unit/workflow/test_async_graph.py`
- Replace/update: `tests/unit/workflow/test_interrupt.py`

**Interfaces:**
- `build_graph(*, policy, risk_classifier, routing_snapshots, rag_answerer, toolbox, checkpointer=None, max_planned_tasks=4) -> CompiledGraph`.
- Graph state adds `inbound_message_id`, `claim_token`, `routing_snapshot`, `final_routing`, and `escalation_intent`.
- `prepare_escalation(state) -> EscalationIntent` is pure with respect to cases, reviews, outbox, and mailbox.

- [ ] **Step 1: Write graph route tests against fakes.**

Test a deterministic confidentiality request, deterministic external action, indirect semantic-risk request, valid semantic routine request, uncertain result, classifier failure, insufficient evidence, conflicting evidence, citation failure, greeting, clarification, and routine cited answer. Assert generator call count is zero on every pre-retrieval review route and the graph never emits `__interrupt__`.

- [ ] **Step 2: Run the graph tests and confirm legacy interrupt behavior fails the new contract.**

Run: `pytest tests/unit/workflow/test_async_graph.py tests/unit/workflow/test_interrupt.py -q`

Expected: new tests fail and existing interrupt tests identify the old behavior that must be replaced.

- [ ] **Step 3: Add intake snapshot loading and deterministic triage.**

At intake, load an existing routing snapshot by inbound identity. If present, route from it and skip policy/classifier calls. If absent, run deterministic triage, then invoke the semantic classifier only for substantive no-match candidates. Persist the validated snapshot before retrieval, generation, or escalation side effects, using the active claim token.

- [ ] **Step 4: Replace live review with escalation intent.**

Route hard-rule and semantic review/uncertain results through evidence-only retrieval and `prepare_escalation`. Route semantic routine results through plan, fan-out/join, compose, and verification. Route invalid or unavailable classification to a safe failure state. Remove `interrupt()` and case creation from the graph; keep checkpointing only as optional computational state, never as the review queue.

- [ ] **Step 5: Add post-retrieval gates.**

Map insufficient evidence, conflicting evidence, missing citations, and citation-grounding failure to their exact durable categories. Discard the generated draft for every failed gate and return an escalation intent with evidence status. Preserve selected, insufficient, conflicting, or unavailable evidence for the reviewer.

- [ ] **Step 6: Run graph, lint, and type checks and commit.**

Run: `pytest tests/unit/workflow/test_async_graph.py tests/unit/workflow/test_policy.py -q && ruff check src/pwc_support/workflow tests/unit/workflow && mypy src/pwc_support/workflow`

```bash
git add src/pwc_support/workflow src/pwc_support/domain/state.py tests/unit/workflow
git commit -m "feat: route review asynchronously without graph interrupts"
```

### Task 6: Client service, asynchronous review service, and dispatcher

**Files:**
- Modify: `src/pwc_support/services/client_support.py`
- Create: `src/pwc_support/services/review.py`
- Modify: `src/pwc_support/bootstrap.py`
- Create: `tests/unit/services/test_async_client_support.py`
- Create: `tests/unit/services/test_review_service.py`

**Interfaces:**
- `ClientSupportService.submit(..., provider_message_id: str | None = None) -> WorkflowRun`.
- `ClientSupportService.retry_submission(original: WorkflowRun) -> WorkflowRun` creates a new provider-message identity while preserving body, contact, and conversation.
- `ReviewService.decide(*, review_id: UUID, decision_id: UUID, expected_version: int, reviewer_id: str, kind: ReviewDecisionKind, reviewed_text: str | None = None, reason: str | None = None) -> ReviewDecisionResult`.
- `OutboxDispatcher.dispatch_once(*, worker_id: str) -> DeliveryReceipt | None`.

- [ ] **Step 1: Write service lifecycle tests.**

Test routine chat no case, deterministic and semantic case IDs returned before request completion, one case plus review, simulated email acknowledgement, chat inline acknowledgement, service-failure email, new-identity chat retry, duplicate identity replay, reviewer send/ownership/reject, stale decisions, and delivery retry.

- [ ] **Step 2: Run service tests and confirm failure against the interrupt-based service.**

Run: `pytest tests/unit/services/test_async_client_support.py tests/unit/services/test_review_service.py -q`

Expected: FAIL because the current service persists review records after an interrupt and resumes the graph instead of committing an asynchronous case outcome.

- [ ] **Step 3: Implement leased submission and completion.**

Claim the inbound identity before invoking the graph. Pass the claim token and inbound identity into graph state. Renew at stage boundaries. On an existing completed identity return its stored outcome; on an unexpired processing identity return `IN_PROGRESS`; on an expired identity recover using its persisted routing snapshot. Complete through the repository transaction and dispatch one best-effort outbox attempt after commit.

- [ ] **Step 4: Implement review decisions without graph resume.**

`send_response` requires non-empty reviewer text, persists the decision and response version, moves the case to `delivery_pending`, and inserts one reviewed-response outbox row. `take_ownership` moves to `human_owned`; `reject` moves to `rejected`. No decision calls `graph.invoke(Command(resume=...))`.

- [ ] **Step 5: Implement explicit channel retry semantics.**

Render classifier or RAG failure as inline chat failure or a single simulated-email `service_failure` message. A chat retry generates a new provider-message UUID; a Streamlit rerun retains the original UUID and replays the stored failure. Do not create a case for a classifier or infrastructure outage.

- [ ] **Step 6: Run checks and commit.**

Run: `pytest tests/unit/services/test_async_client_support.py tests/unit/services/test_review_service.py -q && ruff check src/pwc_support/services src/pwc_support/bootstrap.py tests/unit/services && mypy src/pwc_support/services src/pwc_support/bootstrap.py`

```bash
git add src/pwc_support/services src/pwc_support/bootstrap.py tests/unit/services
git commit -m "feat: add asynchronous review decisions and dispatch"
```

### Task 7: SQLite simulated mailbox and Streamlit client/reviewer workflows

**Files:**
- Modify: `src/pwc_support/adapters/simulated_mailbox.py`
- Modify: `app.py`
- Modify: `src/pwc_support/bootstrap.py`
- Create: `tests/unit/services/test_sqlite_mailbox.py`
- Create: `tests/ui/test_async_contracts.py`

**Interfaces:**
- `SimulatedMailbox.receive(...) -> IncomingMessage` persists inbound identity and thread.
- `SimulatedMailbox.deliver(message: OutboundMessage) -> DeliveryReceipt` inserts by delivery key and returns the existing receipt for an identical retry.
- `SimulatedMailbox.list_messages(*, recipient: str, thread_id: str) -> list[MailboxMessage]` filters the simulated mailbox.

- [ ] **Step 1: Write mailbox idempotency and filtering tests.**

Assert same-key same-payload returns one receipt, same-key different-payload raises an idempotency conflict, thread and recipient are preserved, and one recipient cannot see another recipient's messages.

- [ ] **Step 2: Replace the JSON operational path with SQLite delivery.**

Keep the JSON file readable only as legacy demo data if needed, but use `mailbox_messages` for all new writes and reads. Store message kind, delivery key, payload hash, provider IDs, recipient, and thread.

- [ ] **Step 3: Update the client chat.**

Add a required local-only simulated reply email field. Show routine cited answers inline, case ID and awaiting-review status for escalations, and a filtered `My simulated mailbox` panel. Add a retry button that creates a new provider identity only for classifier/RAG failures. Do not expose policy categories, internal rationale, raw chunks, or reviewer controls to the client.

- [ ] **Step 4: Update the simulated-email tab.**

Preserve sender and provider thread, show immediate acknowledgement or service failure, list only the selected sender/thread, and expose retry for failed non-resolving messages. Automatic answers, acknowledgements, reviewed responses, and service failures must display their message kind and timestamp.

- [ ] **Step 5: Replace the reviewer tab.**

Read pending records from `ReviewRepository`, never from LangGraph checkpoints. Show case ID, status, recipient/thread, original enquiry, categories, evidence state, citations, routing provenance summary, and response version. Keep the response field blank for every case. Implement conditional `Send reviewed response`, `Take ownership`, `Reject`, and retry controls with explicit receipts or retryable errors.

- [ ] **Step 6: Run UI-adjacent tests and commit.**

Run: `pytest tests/unit/services/test_sqlite_mailbox.py tests/unit/services/test_async_client_support.py -q && ruff check app.py src/pwc_support/adapters tests/unit/services && mypy src/pwc_support/adapters`

```bash
git add app.py src/pwc_support/adapters src/pwc_support/bootstrap.py tests/unit/services tests/ui
git commit -m "feat: expose durable async review in the demo UI"
```

### Task 8: Frozen semantic evaluation, load cohorts, documentation, and release verification

**Files:**
- Create: `eval/risk_routing.jsonl`
- Modify: `eval/final.jsonl`
- Modify: `eval/load_workload.jsonl`
- Modify: `scripts/run_evaluation.py`
- Modify: `scripts/run_load.py`
- Modify: `README.md`
- Modify: `docs/audits/2026-09-03-remediation-verification.md`
- Create: `tests/integration/test_async_lifecycle.py`

**Interfaces:**
- Risk dataset records `id`, `body`, `gold_route`, `gold_categories`, `hard_rule_expected`, `annotation_rationale`, and `split`.
- Evaluation output records confusion matrices, Wilson 95% intervals, rule baseline, hybrid result, invalid/timeout counts, queue volume, and pinned artifact identifiers.
- Load output records separate cohorts for classifier-skipped, classifier-invoked routine, classifier-invoked review, reviewer delivery, and queue wait.

- [ ] **Step 1: Create and freeze the semantic-risk dataset.**

Add 80 manually reviewed examples: 40 development and 40 frozen final, with at least 24 final review-positive cases (four per semantic category) and 16 routine negatives. Include indirect paraphrases, negation, quoted risky text, multi-intent requests, prompt injection, ambiguous language, and near-boundary routine questions. Hash the file before final evaluation.

- [ ] **Step 2: Extend the evaluation runner.**

Run the rule-only baseline and the hybrid classifier on the frozen split without prompt tuning. Report exact confusion matrices and Wilson 95% intervals. Enforce the declared gates: 100% deterministic hard-rule recall, at least 90% recall among review examples missed by hard rules, at most 20% routine false-positive escalation, and at most 5% invalid structured outputs. Record model digest, options, prompt/taxonomy/schema versions, dataset hash, and seed.

- [ ] **Step 3: Add multi-step async lifecycle scenarios.**

Extend final scenarios to submit, optional process rebuild, reviewer decision, dispatch, and expected record counts. Include routine no-case, hard-rule case, indirect semantic case, uncertain, unavailable/service-failure, evidence-insufficient, send, ownership, reject, duplicate input, stale decision, acknowledgement failure, and exactly-once delivery retry.

- [ ] **Step 4: Add real-runtime integration coverage.**

Use isolated SQLite paths and fakes for deterministic repository assertions, then run the relevant Ollama/Chroma scenarios when local services are available. Assert sensitive generator-call count, routing provenance, case/review/outbox counts, mailbox thread, and separate machine timings. Do not treat unit fakes as real-runtime evidence.

- [ ] **Step 5: Split load cohorts and latency metrics.**

Measure greeting/clarification, hard-rule case, classifier-routine RAG, classifier-review case, and reviewer delivery separately. Report p50/p95/p99, throughput, errors, invalid/timeout rates, `risk_classification_ms`, `live_submit_ms`, `queue_wait_ms`, and delivery latency. Keep queue wait out of live request latency.

- [ ] **Step 6: Update documentation and run the full verification gate.**

Document that current tools remain typed custom Python tools, not LangChain `@tool` bindings; document the classifier's escalation-only authority, asynchronous case ID behavior, simulated mailbox limitations, and exact retry semantics. Run:

```bash
pytest -q
ruff check .
mypy src
python scripts/run_evaluation.py --dataset eval/final.jsonl
python scripts/run_load.py --workload eval/load_workload.jsonl
git diff --check
```

Expected: all tests and static checks pass; evaluation and load artifacts include the pinned configuration and separate routing/lifecycle evidence. Record any unavailable Ollama/Chroma run as unavailable rather than claiming a real-runtime pass.

- [ ] **Step 7: Commit the verified implementation documentation.**

```bash
git add eval scripts/run_evaluation.py scripts/run_load.py README.md docs/audits/2026-09-03-remediation-verification.md tests/integration/test_async_lifecycle.py
git commit -m "test: evaluate async review routing and lifecycle"
```

## Plan Self-Review

- Domain contracts precede every consumer and define category cardinality, delivery kinds, leases, and provenance.
- Policy and classifier tests prove monotonic escalation before graph work begins.
- Storage tests prove the claim-token, snapshot-winner, atomic completion, acknowledgement, and outbox contracts before service/UI work.
- Graph tests prove no interrupt, no sensitive generation, post-RAG case routing, and safe classifier failure.
- Service tests prove immediate case IDs, duplicate identity behavior, new-identity chat retry, reviewer CAS, and exactly-once delivery.
- UI work reads durable repositories and filters mailbox messages by recipient/thread.
- Evaluation covers both the assignment's bounded end-to-end scenarios and a separate frozen semantic classifier set.
- No task relies on a placeholder, an unversioned policy, mutable logs, or a LangGraph checkpoint as the review queue.

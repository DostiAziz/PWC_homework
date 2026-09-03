# Agentic Specialist Routing and Order Cancellation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace static retail forms and keyword branches with a classifier-driven, multi-intent LangGraph supervisor that delegates to reusable specialist agents, preserves RAG, and supports memory-aware order cancellation review.

**Architecture:** The main graph classifies each message into one or more validated tasks, fans out independent tasks to reusable ProductAgent, OrderAgent, ReturnRefundAgent, and RagAgent subgraphs, then joins typed results. OrderAgent owns order lookup, clarification memory, status and payment inspection, cancellation-policy evaluation, and review-packet preparation; the main workflow owns customer responses and durable human-review persistence.

**Tech Stack:** Python 3.12, LangGraph 1.2.11, Pydantic 2.13.5, local Ollama `gpt-oss:20b`, Chroma plus SQLite FTS5 for RAG, SQLite for synthetic retail operations and conversation memory, Streamlit 1.63.0, pytest, Ruff, and mypy.

**Spec:** [docs/superpowers/specs/2026-09-03-agentic-specialist-routing-design.md](../specs/2026-09-03-agentic-specialist-routing-design.md)

## Global Constraints

- The client experience is one natural-language conversation; do not add product, order, cancellation, return, or recommendation forms.
- The classifier may create multiple tasks, but every task must use a closed `SupportIntent` and a registered specialist.
- Order status and cancellation use the same `OrderAgent`; status inspection is an internal prerequisite of cancellation.
- The existing contextual hybrid RAG graph remains the `RagAgent` implementation for unstructured policy and general-knowledge questions.
- Product, price, stock, offer, order, payment-state, and fulfilment facts must come from bounded database tools, never generated text.
- Cancellation is always assessed first and sent to human review; no automatic cancellation tool is exposed to the agent.
- The main workflow creates the case and review request after receiving a typed specialist result; specialists do not write cases or outbox rows.
- A missing order ID produces clarification. The first lookup miss asks for correction. The second lookup miss returns `ORDER_NOT_FOUND` through the main workflow and creates no review case.
- Conversation memory is scoped by `(conversation_id, client_id)` and must not retain card numbers, secrets, or unrestricted payment references.
- Customer-visible responses must not expose internal policy rule text, fraud scores, raw SQL, or private model reasoning.
- Every task and specialist node records a correlation ID, task ID, node timing, safe tool summary, and terminal status.
- Use TDD for every behavior change. Before claiming completion, run focused tests, full `pytest`, Ruff, and strict mypy.

---

## File and responsibility map

Create the following focused modules:

- `src/pwc_support/agents/__init__.py`: public specialist exports.
- `src/pwc_support/agents/contracts.py`: shared routing, specialist, memory, policy, and review-packet models.
- `src/pwc_support/agents/classifier.py`: structured multi-intent classifier adapter and plan validation.
- `src/pwc_support/agents/order.py`: reusable OrderAgent graph and its bounded tool-loop state machine.
- `src/pwc_support/agents/product.py`: reusable ProductAgent graph for read-only catalogue work.
- `src/pwc_support/agents/returns.py`: reusable ReturnRefundAgent graph for return/refund assessment.
- `src/pwc_support/agents/rag.py`: adapter from routed knowledge tasks to the existing RAG subgraph.
- `src/pwc_support/storage/conversation_state.py`: durable conversation-specialist memory repository.

Modify the following existing modules:

- `src/pwc_support/domain/models.py`: add closed intent, specialist, investigation, policy, and action contracts; preserve unrelated public models.
- `src/pwc_support/domain/errors.py`: add a dedicated intent-classification-unavailable error.
- `src/pwc_support/domain/state.py`: add supervisor task-plan, active-agent-memory, specialist-result, and joined-result slices with reducers.
- `src/pwc_support/storage/database.py`: initialize conversation-memory and order-investigation/cancellation schema additions safely for existing SQLite files.
- `src/pwc_support/storage/retail_schema.sql`: add payment, fulfilment, shipping, version, and cancellation-action fields/tables.
- `src/pwc_support/storage/retail_repositories.py`: expose typed investigation and optimistic cancellation operations.
- `src/pwc_support/workflow/retail_tools.py`: expose bounded typed order-investigation tools and keep automatic cancellation mutation unavailable.
- `src/pwc_support/workflow/retail_actions.py`: apply reviewer-approved cancellation actions idempotently.
- `src/pwc_support/services/review.py`: support cancellation approval/rejection and preserve action-before-delivery ordering.
- `src/pwc_support/services/client_support.py`: load and save conversation-specialist memory around each graph invocation.
- `src/pwc_support/workflow/graph.py`: replace keyword retail branches with classifier, active-agent resume, specialist fan-out, join, and main-workflow handoff.
- `src/pwc_support/bootstrap.py`: construct classifier, specialist subgraphs, conversation memory repository, and dependencies explicitly.
- `src/pwc_support/config.py`: add intent-classifier settings, specialist loop limits, and conversation-memory limits.
- `app.py`: remove static retail forms and present one conversational client surface plus operational trace and internal review information.
- `README.md`: document the new agent topology, memory behavior, cancellation review boundary, and test commands.

Update or create focused tests:

- `tests/unit/agents/test_contracts.py`
- `tests/unit/agents/test_classifier.py`
- `tests/unit/agents/test_order_agent.py`
- `tests/unit/agents/test_product_agent.py`
- `tests/unit/agents/test_return_refund_agent.py`
- `tests/unit/agents/test_rag_agent.py`
- `tests/unit/storage/test_conversation_state.py`
- `tests/unit/storage/test_retail_repositories.py`
- `tests/unit/workflow/test_supervisor_graph.py`
- `tests/unit/workflow/test_retail_tools.py`
- `tests/unit/services/test_cancellation_review.py`
- `tests/integration/test_agentic_customer_journey.py`
- `tests/ui/test_retail_contracts.py`

---

### Task 1: Add typed routing, specialist, memory, and cancellation contracts

**Files:**
- Create: `src/pwc_support/agents/__init__.py`
- Create: `src/pwc_support/agents/contracts.py`
- Modify: `src/pwc_support/domain/models.py:53-75, 92-106, 342-367`
- Modify: `src/pwc_support/domain/state.py:9-39`
- Test: `tests/unit/agents/test_contracts.py`
- Modify: `tests/unit/domain/test_retail_models.py`

**Interfaces:**
- Produces `SupportIntent`, `SpecialistName`, `SpecialistStatus`, `RoutingTask`, `RoutingDecision`, `PolicyFinding`, `OrderInvestigation`, `CancellationReviewPacket`, `CancellationExecutionResult`, `SpecialistResult`, and `ConversationMemory`.
- Produces `SupportState` slices named `routing_decision`, `routing_tasks`, `active_specialist`, `conversation_memory`, `specialist_results`, and `joined_response_parts`.
- Extends `ProposedAction.action_type` with `retail_cancellation` and `ReviewDecisionKind` with `APPROVE_CANCELLATION` and `REJECT_CANCELLATION`.

- [ ] **Step 1: Write failing contract tests.**

```python
def test_routing_task_accepts_multiple_specialist_intents() -> None:
    task = RoutingTask(
        task_id="order-1",
        intent=SupportIntent.ORDER_CANCELLATION,
        specialist=SpecialistName.ORDER,
        user_text="Cancel order ORD-1001 if it has not shipped",
        entities={"order_id": "ORD-1001"},
        depends_on=(),
    )
    assert task.specialist is SpecialistName.ORDER


def test_cancellation_review_packet_requires_verified_order_facts() -> None:
    packet = CancellationReviewPacket(
        conversation_id=uuid4(),
        client_id="CUS-1001",
        original_request="Cancel my order",
        order=OrderInvestigation(
            order_id="ORD-1001",
            customer_id="CUS-1001",
            status="processing",
            total=Decimal("129.00"),
            currency="EUR",
            payment_status="paid",
            fulfilment_status="processing",
            shipped_at=None,
            version=1,
            items=(),
        ),
        policy_findings=(PolicyFinding(rule_id="not_shipped", passed=True, message="Order has not shipped"),),
        recommended_action="human_review",
    )
    assert packet.order.order_id == "ORD-1001"
```

- [ ] **Step 2: Run the focused tests and verify they fail.**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/agents/test_contracts.py -q`

Expected: FAIL because the new contract types do not exist.

- [ ] **Step 3: Implement the models.**

Use closed `StrEnum` values for `SupportIntent`, `SpecialistName`, and `SpecialistStatus`. Make
`RoutingTask.depends_on` a tuple of task IDs, `entities` a bounded `dict[str, str]`, and apply
`extra="forbid"` through the existing `DomainModel`. Make `CancellationReviewPacket.order` an
`OrderInvestigation`, so a review packet cannot be created from an unverified order ID alone.

Define `ConversationMemory` with `conversation_id`, `client_id`, `active_specialist`,
`active_task_id`, `state_json`, `version`, and `updated_at`. Validate that a memory record with an
active specialist has an active task ID.

Define `CancellationExecutionResult` with the order ID, review ID, final status, and a `replayed`
flag so approval retries can be distinguished from the first successful mutation.

- [ ] **Step 4: Add state reducers and action enum values.**

Add append/merge behavior for specialist results keyed by `task_id`; reject a conflicting second
result for the same task. Add the cancellation action literal and approval/rejection enum values.
Keep existing return/refund values unchanged.

- [ ] **Step 5: Run focused tests and static checks.**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/agents/test_contracts.py tests/unit/domain/test_retail_models.py -q`

Expected: PASS. Then run: `ruff check src/pwc_support/agents src/pwc_support/domain tests/unit/agents tests/unit/domain`.

- [ ] **Step 6: Commit the contracts.**

```bash
git add src/pwc_support/agents src/pwc_support/domain/models.py src/pwc_support/domain/state.py tests/unit/agents tests/unit/domain/test_retail_models.py
git commit -m "feat: add specialist routing contracts"
```

---

### Task 2: Extend the retail database for order investigation and cancellation

**Files:**
- Modify: `src/pwc_support/storage/retail_schema.sql:16-39`
- Modify: `src/pwc_support/storage/database.py:160-200`
- Modify: `src/pwc_support/storage/retail_repositories.py:102-128`
- Modify: `src/pwc_support/domain/models.py:120-127`
- Modify: `scripts/seed_retail_data.py`
- Test: `tests/unit/storage/test_retail_repositories.py`
- Test: `tests/unit/storage/test_retail_schema.py`

**Interfaces:**
- Produces `OrderRepository.investigate(order_id: str, customer_id: str) -> OrderInvestigation | None`.
- Produces `OrderRepository.cancel(order_id: str, customer_id: str, *, expected_version: int, review_id: str) -> CancellationExecutionResult`.
- Produces exact order facts for status, total, currency, payment state, fulfilment state, shipment timestamp, version, and items.

- [ ] **Step 1: Write failing schema and repository tests.**

```python
def test_investigation_returns_payment_and_fulfilment_facts(tmp_path: Path) -> None:
    database = _seed_database(tmp_path, status="processing", payment_status="paid", fulfilment_status="processing")
    investigation = OrderRepository(database).investigate("ORD-1", "CUS-1")
    assert investigation is not None
    assert investigation.payment_status == "paid"
    assert investigation.fulfilment_status == "processing"
    assert investigation.version == 1


def test_cancel_requires_expected_order_version(tmp_path: Path) -> None:
    database = _seed_database(tmp_path, status="processing")
    repository = OrderRepository(database)
    with pytest.raises(ValueError, match="version conflict"):
        repository.cancel("ORD-1", "CUS-1", expected_version=2, review_id="review-1")


def test_cancel_is_idempotent_for_same_review(tmp_path: Path) -> None:
    database = _seed_database(tmp_path, status="processing")
    repository = OrderRepository(database)
    first = repository.cancel("ORD-1", "CUS-1", expected_version=1, review_id="review-1")
    replay = repository.cancel("ORD-1", "CUS-1", expected_version=1, review_id="review-1")
    assert first.status == "cancelled"
    assert replay.status == "cancelled"
```

- [ ] **Step 2: Run the tests to verify the schema and methods fail.**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/storage/test_retail_repositories.py tests/unit/storage/test_retail_schema.py -q`

Expected: FAIL because the new columns and repository methods are absent.

- [ ] **Step 3: Add schema fields with safe defaults.**

Extend `orders` with `payment_status`, `fulfilment_status`, `shipped_at`, `cancelled_at`, and
`version`. Add `order_cancellation_actions` with a unique `review_id` and `idempotency_key`, the
order/customer identity, status, and timestamps. Add additive `_add_missing_column` calls in
`Database.initialize()` for existing databases. Use explicit column lists in every seed and test
insert so future schema fields cannot silently reorder fixture values.

- [ ] **Step 4: Implement typed investigation and optimistic cancellation.**

Make `investigate()` query by both order ID and customer ID. Return `None` for both an unknown order
and an order belonging to another customer. Make `cancel()` insert-or-reuse the action record,
update the order only when the supplied version matches, set `status='cancelled'`, set
`cancelled_at`, increment `version`, and write a retail audit event. Return a typed result with
`replayed=True` for the same review replay.

- [ ] **Step 5: Update seed data and fixtures.**

Seed at least one processing and unpaid order, one paid and unshipped order, one shipped order, and
one delivered order. Keep `ORD-1001` as the standard client fixture. Add payment and fulfilment
values to all existing test fixture rows.

- [ ] **Step 6: Run focused tests and commit.**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/storage/test_retail_repositories.py tests/unit/storage/test_retail_schema.py -q`

Expected: PASS. Commit:

```bash
git add src/pwc_support/storage/retail_schema.sql src/pwc_support/storage/database.py src/pwc_support/storage/retail_repositories.py src/pwc_support/domain/models.py scripts/seed_retail_data.py tests/unit/storage/test_retail_repositories.py tests/unit/storage/test_retail_schema.py
git commit -m "feat: add order investigation and cancellation state"
```

---


### Task 3: Persist durable conversation-specialist memory

**Files:**
- Create: `src/pwc_support/storage/conversation_state.py`
- Modify: `src/pwc_support/storage/database.py:160-200`
- Modify: `src/pwc_support/services/client_support.py:40-110`
- Modify: `src/pwc_support/config.py:1-170`
- Test: `tests/unit/storage/test_conversation_state.py`

**Interfaces:**
- Produces `ConversationStateRepository.load(conversation_id: UUID, client_id: str) -> ConversationMemory | None`.
- Produces `ConversationStateRepository.save(memory: ConversationMemory, *, expected_version: int | None) -> ConversationMemory`.
- Produces `ConversationStateRepository.clear(conversation_id: UUID, client_id: str, *, expected_version: int) -> None`.

- [ ] **Step 1: Write failing persistence tests.**

```python
def test_memory_survives_repository_reconstruction(tmp_path: Path) -> None:
    database = Database(tmp_path / "support.sqlite")
    first = ConversationStateRepository(database)
    memory = ConversationMemory(
        conversation_id=uuid4(),
        client_id="CUS-1",
        active_specialist=SpecialistName.ORDER,
        active_task_id="order-1",
        state_json={"phase": "awaiting_order_id", "attempts": 0},
        version=0,
        updated_at=datetime.now(timezone.utc),
    )
    saved = first.save(memory, expected_version=None)
    loaded = ConversationStateRepository(database).load(saved.conversation_id, "CUS-1")
    assert loaded is not None
    assert loaded.state_json["phase"] == "awaiting_order_id"


def test_memory_is_isolated_by_client_id(tmp_path: Path) -> None:
    database = Database(tmp_path / "support.sqlite")
    repository = ConversationStateRepository(database)
    memory = _memory(client_id="CUS-1")
    repository.save(memory, expected_version=None)
    assert repository.load(memory.conversation_id, "CUS-2") is None


def test_stale_memory_update_is_rejected(tmp_path: Path) -> None:
    database = Database(tmp_path / "support.sqlite")
    repository = ConversationStateRepository(database)
    saved = repository.save(_memory(), expected_version=None)
    with pytest.raises(ValueError, match="memory version conflict"):
        repository.save(saved.model_copy(update={"version": 0}), expected_version=0)
```

Define `_memory(client_id: str = "CUS-1") -> ConversationMemory` in the test module and import
`datetime`, `timezone`, `Path`, `uuid4`, and `pytest`; the helper must create a fresh conversation ID
and an active OrderAgent state.

- [ ] **Step 2: Run the focused tests and verify they fail.**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/storage/test_conversation_state.py -q`

Expected: FAIL because the repository and table do not exist.

- [ ] **Step 3: Add the conversation-memory table and repository.**

Create `conversation_agent_state` with a composite primary key on `(conversation_id, client_id)`,
an integer version, JSON state, active specialist/task fields, update timestamp, and a bounded
state payload. Serialize only the typed specialist state needed to resume clarification or review
preparation. Reject payloads above `settings.conversation_memory_max_chars` and reject compare-and-
swap updates whose stored version differs from `expected_version`. Use parameterized SQL and the
existing database transaction helper.

- [ ] **Step 4: Integrate load and save around the main graph invocation.**

Load memory by conversation and client before invoking the supervisor. Pass it as the initial
`SupportState.conversation_memory` slice. Save only the explicit `conversation_memory_update`
returned by the graph, using the version observed at load time. Clear the active specialist/task
when a task reaches a terminal result, while retaining a short, redacted summary for the next
turn. Preserve the inbound message and existing idempotency behavior.

- [ ] **Step 5: Add configuration and run checks.**

Add `conversation_memory_max_chars=12000`, `specialist_max_steps=8`, and the corresponding
environment-variable settings using the existing `Settings` conventions. Run:

`PYTHONPATH=src .venv/bin/python -m pytest tests/unit/storage/test_conversation_state.py tests/unit/services/test_client_support.py -q`

Expected: PASS. Commit:

```bash
git add src/pwc_support/storage/conversation_state.py src/pwc_support/storage/database.py src/pwc_support/services/client_support.py src/pwc_support/config.py tests/unit/storage/test_conversation_state.py tests/unit/services/test_client_support.py
git commit -m "feat: persist specialist conversation memory"
```

---

### Task 4: Add structured multi-intent classification and task planning

**Files:**
- Create: `src/pwc_support/agents/classifier.py`
- Modify: `src/pwc_support/agents/contracts.py`
- Modify: `src/pwc_support/domain/errors.py`
- Modify: `src/pwc_support/config.py:1-170`
- Test: `tests/unit/agents/test_classifier.py`

**Interfaces:**
- Produces `IntentClassifier.classify(message: str) -> RoutingDecision`.
- Produces `OllamaIntentClassifier`, which uses `OllamaGenerator.structured` with a Pydantic schema.
- Produces `validate_routing_decision(decision: RoutingDecision, max_tasks: int) -> RoutingDecision`.

- [ ] **Step 1: Write failing classifier tests.**

```python
def test_classifier_preserves_compound_order_and_knowledge_intents() -> None:
    decision = _classifier().classify(
        "Can you tell me whether order ORD-1001 has shipped and explain the cancellation policy?"
    )
    assert [task.specialist for task in decision.tasks] == [
        SpecialistName.ORDER,
        SpecialistName.RAG,
    ]
    assert decision.tasks[1].depends_on == ()


def test_validator_rejects_unknown_specialist_and_dependency_cycle() -> None:
    with pytest.raises(ValueError, match="registered specialist"):
        validate_routing_decision(_decision_with_unknown_specialist(), max_tasks=4)
    with pytest.raises(ValueError, match="cycle"):
        validate_routing_decision(_decision_with_cycle(), max_tasks=4)


def test_classifier_failure_returns_safe_unavailable_error() -> None:
    with pytest.raises(IntentClassificationUnavailable):
        _failing_classifier().classify("Where is my order?")
```

Define `_classifier()`, `_failing_classifier()`, `_decision_with_unknown_specialist()`, and
`_decision_with_cycle()` as test helpers using fake structured-model collaborators. The fakes must
return Pydantic data or raise the same exception as the Ollama adapter, so these tests do not need a
running model.

- [ ] **Step 2: Run focused tests to verify they fail.**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/agents/test_classifier.py -q`

Expected: FAIL because the classifier contracts and adapter do not exist.

- [ ] **Step 3: Define the closed classification schema.**

Use the following intents: `PRODUCT_SEARCH`, `PRODUCT_RECOMMENDATION`, `ORDER_STATUS`,
`ORDER_CANCELLATION`, `RETURN_REQUEST`, `REFUND_REQUEST`, and `KNOWLEDGE_QUERY`. Map product
search/recommendation to ProductAgent, order status/cancellation to OrderAgent, return/refund to
ReturnRefundAgent, and policy/general questions to RagAgent. Each task contains normalized entities,
the original user text, a bounded task ID, and dependency IDs. Reject unknown values, duplicate task
IDs, self-dependencies, cycles, missing dependency IDs, and plans above `settings.max_planned_tasks`.

- [ ] **Step 4: Implement the Ollama adapter with a safe failure boundary.**

Call `OllamaGenerator.structured` at temperature zero with a system prompt that describes only the
closed schema and instructs the model to classify untrusted user text, not to call tools, mutate
orders, or approve reviews. Delimit the user message as data. Normalize order IDs with the existing
domain conventions, but leave ambiguous IDs for OrderAgent clarification. Add
`IntentClassificationUnavailable` in `domain/errors.py`; the main workflow will return a safe
clarification/error response and an operational event rather than inventing a route.

- [ ] **Step 5: Run tests, static checks, and commit.**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/agents/test_classifier.py -q`

Expected: PASS. Then run: `ruff check src/pwc_support/agents src/pwc_support/domain/errors.py tests/unit/agents/test_classifier.py`.

```bash
git add src/pwc_support/agents src/pwc_support/domain/errors.py src/pwc_support/config.py tests/unit/agents/test_classifier.py
git commit -m "feat: add structured multi-intent classifier"
```

---

### Task 5: Build the memory-aware OrderAgent subgraph

**Files:**
- Create: `src/pwc_support/agents/order.py`
- Modify: `src/pwc_support/workflow/retail_tools.py:1-180`
- Modify: `src/pwc_support/domain/models.py:120-160`
- Test: `tests/unit/agents/test_order_agent.py`
- Test: `tests/unit/workflow/test_retail_tools.py`

**Interfaces:**
- Produces `build_order_agent(*, tools: OrderToolbox, planner: OrderPlanner, max_steps: int) -> CompiledStateGraph`.
- Produces `OllamaOrderPlanner.next_step(state: OrderAgentState) -> OrderNextStep`.
- Produces an `OrderAgentState` with request, messages, order ID, lookup attempts, investigation,
  policy findings, review packet, phase, and typed terminal result.

- [ ] **Step 1: Write failing agent behavior tests.**

```python
def test_missing_order_id_requests_it_without_lookup() -> None:
    result = invoke_order_agent("I want to cancel my order", memory=_memory())
    assert result.status is SpecialistStatus.NEEDS_INFORMATION
    assert result.customer_message == "Please provide your order number so I can check it."
    assert result.tool_calls == ()


def test_first_lookup_miss_requests_correction_and_persists_attempt() -> None:
    result = invoke_order_agent("ORD-9999", memory=_memory(phase="awaiting_order_id"))
    assert result.status is SpecialistStatus.CORRECTION_REQUESTED
    assert result.memory_update.state_json["lookup_attempts"] == 1


def test_second_lookup_miss_returns_not_found_without_review() -> None:
    result = invoke_order_agent("ORD-8888", memory=_memory(phase="awaiting_correction", attempts=1))
    assert result.status is SpecialistStatus.ORDER_NOT_FOUND
    assert result.review_packet is None


def test_found_cancellable_order_returns_review_packet_without_mutation() -> None:
    result = invoke_order_agent("Please cancel ORD-1001", memory=_memory())
    assert result.status is SpecialistStatus.REVIEW_REQUIRED
    assert result.review_packet is not None
    assert result.review_packet.order.status == "processing"
    assert result.tools.cancel_call_count == 0
```

Define `_memory(phase: str = "new", attempts: int = 0)`, `invoke_order_agent(...)`, and a
`fake_order_tools` fixture in the test module. The fake repository must record every tool call and
return deterministic investigations for `ORD-1001`, `None` for `ORD-9999` and `ORD-8888`, and no
mutation method exposed through the agent toolbox.

- [ ] **Step 2: Run focused tests and verify they fail.**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/agents/test_order_agent.py tests/unit/workflow/test_retail_tools.py -q`

Expected: FAIL because the reusable OrderAgent and investigation tools do not exist.

- [ ] **Step 3: Expose only bounded read tools.**

Create an `OrderToolbox` protocol exposing `lookup_order(order_id, customer_id)` and
`inspect_order(order_id, customer_id)`. Both must call the repository with customer scope and
return typed facts. Do not include `cancel_order` in the agent toolbox. Keep product and return
tools available to their own specialists, but stop the main graph from directly selecting tools.

- [ ] **Step 4: Implement the bounded OrderAgent state machine.**

Implement nodes named `merge_memory`, `understand_request`, `collect_order_id`, `lookup_order`,
`inspect_order`, `evaluate_cancellation_policy`, `prepare_cancellation_review`, and
`return_to_main_workflow`. The planner may select only the next node from this closed set and is
bounded by `max_steps`. `collect_order_id` asks for a missing ID. `lookup_order` records an attempt;
the first miss stores `awaiting_correction` and asks the customer to review the number. The second
miss returns `ORDER_NOT_FOUND`, says that no information is related to that order number, clears
the active specialist, and creates no review packet. A found order is inspected for status, total,
currency, payment, fulfilment, shipping, and version before any policy decision.

- [ ] **Step 5: Evaluate policy and prepare review without mutation.**

Implement explicit policy predicates for shipped/delivered/cancelled state, payment/refund
implication, and any existing return/refund conflict. Return typed `PolicyFinding` values with
stable rule IDs and customer-safe messages. For cancellation, create a `CancellationReviewPacket`
containing verified facts, policy findings, missing/conflicting facts, recommended reviewer action,
and provenance. Never expose raw policy text or private planner reasoning to the customer. Preserve
the exact memory update so the main graph can hand back a clarification or review result.

- [ ] **Step 6: Run focused checks and commit.**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/agents/test_order_agent.py tests/unit/workflow/test_retail_tools.py -q`

Expected: PASS. Then run: `ruff check src/pwc_support/agents/order.py src/pwc_support/workflow/retail_tools.py tests/unit/agents/test_order_agent.py tests/unit/workflow/test_retail_tools.py`.

```bash
git add src/pwc_support/agents/order.py src/pwc_support/workflow/retail_tools.py src/pwc_support/domain/models.py tests/unit/agents/test_order_agent.py tests/unit/workflow/test_retail_tools.py
git commit -m "feat: add memory-aware order specialist"
```

---

### Task 6: Build ProductAgent, ReturnRefundAgent, and RagAgent adapters

**Files:**
- Create: `src/pwc_support/agents/product.py`
- Create: `src/pwc_support/agents/returns.py`
- Create: `src/pwc_support/agents/rag.py`
- Modify: `src/pwc_support/workflow/retail_actions.py:1-180`
- Test: `tests/unit/agents/test_product_agent.py`
- Test: `tests/unit/agents/test_return_refund_agent.py`
- Test: `tests/unit/agents/test_rag_agent.py`

**Interfaces:**
- Produces `build_product_agent(tools: ProductToolbox, planner: ProductPlanner) -> CompiledStateGraph`.
- Produces `build_return_refund_agent(tools: ReturnToolbox) -> CompiledStateGraph`.
- Produces `build_rag_agent(rag_answerer: RagAnswerer) -> CompiledStateGraph`.

- [ ] **Step 1: Write failing adapter tests.**

```python
def test_product_agent_returns_database_facts_for_price_and_offer() -> None:
    result = invoke_product_agent("What is the price and active offer for the travel backpack?")
    assert result.facts["currency"] == "EUR"
    assert result.facts["price_source"] == "product_repository"
    assert result.answer != ""


def test_return_refund_agent_prepares_proposal_without_side_effect() -> None:
    result = invoke_return_agent("I want to return item ITEM-1001")
    assert result.status is SpecialistStatus.REVIEW_REQUIRED
    assert result.action_proposal is not None
    assert result.tools.create_call_count == 0


def test_rag_agent_preserves_citations() -> None:
    result = invoke_rag_agent("What is the cancellation policy?")
    assert result.evidence
    assert all(item.source_id for item in result.evidence)
```

Define `invoke_product_agent`, `invoke_return_agent`, and `invoke_rag_agent` with fake tool and RAG
collaborators in the test modules. The fakes must return typed repository/evidence values and record
any attempted mutation so the tests verify side-effect boundaries.

- [ ] **Step 2: Implement ProductAgent as a bounded read-only tool loop.**

Reuse the existing product repository tools for exact product, price, stock, offer, and
recommendation facts. Let a structured ProductPlanner select only search, get product, inventory,
active offer, or finish. Build the customer answer from returned facts and include uncertainty when
the catalogue has no match. ProductAgent must never invent a price, discount, stock level, or product
identity.

- [ ] **Step 3: Implement ReturnRefundAgent as an assessment subgraph.**

Wrap the existing return/refund eligibility logic with a typed `ReturnToolbox`. It may evaluate an
item and prepare a `ProposedAction`, but it must not create a request, issue money, or write a case.
Return and refund requests become review-ready results for the main workflow, preserving existing
return-window and refund-approval settings.

- [ ] **Step 4: Implement RagAgent as the existing RAG boundary.**

Map a `RoutingTask` to the existing `RagRequest` and invoke `build_rag_graph()` through a small
adapter. Preserve contextual hybrid retrieval, evidence selection, answer citations, and the
existing no-answer behavior. RAG may answer policy/general questions, but it must not provide
order-specific facts that belong to OrderAgent or ProductAgent.

- [ ] **Step 5: Run focused tests and commit.**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/agents/test_product_agent.py tests/unit/agents/test_return_refund_agent.py tests/unit/agents/test_rag_agent.py -q`

Expected: PASS. Commit:

```bash
git add src/pwc_support/agents/product.py src/pwc_support/agents/returns.py src/pwc_support/agents/rag.py src/pwc_support/workflow/retail_actions.py tests/unit/agents/test_product_agent.py tests/unit/agents/test_return_refund_agent.py tests/unit/agents/test_rag_agent.py
git commit -m "feat: add reusable support specialist agents"
```

---

### Task 7: Replace keyword routing with the supervisor and multi-task join

**Files:**
- Modify: `src/pwc_support/workflow/graph.py:1-420`
- Modify: `src/pwc_support/domain/state.py:1-160`
- Modify: `src/pwc_support/services/client_support.py:40-150`
- Modify: `src/pwc_support/bootstrap.py:1-220`
- Test: `tests/unit/workflow/test_supervisor_graph.py`
- Test: `tests/unit/services/test_client_support.py`

**Interfaces:**
- Produces a `build_graph(...)` that accepts classifier, OrderAgent, ProductAgent,
  ReturnRefundAgent, RagAgent, and conversation-memory collaborators explicitly.
- Produces a supervisor path with `intake`, `load_conversation_memory`, `classify_and_plan`,
  `dispatch_specialists`, `join_specialist_results`, `compose_reply`, `prepare_escalation`, and
  `finalise` nodes.

- [ ] **Step 1: Write failing supervisor tests.**

```python
def test_order_message_routes_to_order_agent_not_keyword_tools() -> None:
    result = invoke_supervisor("Where is order ORD-1001?", classifier=_single_order_classifier())
    assert result.specialist_results["order-1"].specialist is SpecialistName.ORDER
    assert result.direct_tool_calls == ()


def test_compound_message_fans_out_order_and_rag_and_joins_both() -> None:
    result = invoke_supervisor(
        "Has ORD-1001 shipped, and what is the cancellation policy?",
        classifier=_compound_order_rag_classifier(),
    )
    assert set(result.specialist_results) == {"order-1", "rag-1"}
    assert result.customer_reply.find("ORD-1001") >= 0
    assert result.rag_results[0].citations


def test_order_correction_resumes_memory_without_losing_main_workflow() -> None:
    first = submit("I want to cancel my order")
    second = submit("ORD-8888", conversation_id=first.conversation_id)
    third = submit("ORD-9999", conversation_id=first.conversation_id)
    assert first.outcome == "clarification_required"
    assert second.outcome == "clarification_required"
    assert third.outcome == "unable_to_answer"
    assert third.review_request_id is None
```

Define `_single_order_classifier`, `_compound_order_rag_classifier`, `invoke_supervisor`, and
`submit` fixtures with fake specialist graphs and an in-memory conversation repository. The fakes
must expose task IDs and typed `SpecialistResult` values so the join assertions exercise real
reducers and routing decisions.

- [ ] **Step 2: Run focused tests and verify the old graph fails them.**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/workflow/test_supervisor_graph.py tests/unit/services/test_client_support.py -q`

Expected: FAIL because the graph currently uses keyword routing and direct retail tools.

- [ ] **Step 3: Add reducer-safe supervisor state.**

Add typed routing decision, task list, active memory, specialist result map, joined response parts,
and review packets to `SupportState`. The specialist result reducer must key by `task_id`, reject
conflicts, and preserve deterministic task order for reply composition. Keep existing RAG and review
state fields compatible with current tests and persisted events.

- [ ] **Step 4: Implement classifier dispatch and LangGraph fan-out.**

Replace keyword-specific retail branches and the direct `plan_work` retail calls with
`classify_and_plan`. If active memory contains an unfinished OrderAgent task, route the new message
to that task for clarification handling before creating a new plan. Otherwise validate the classifier
decision and use LangGraph `Send` to fan out independent tasks to the registered specialist graph.
Respect `depends_on` edges by dispatching only ready tasks and scheduling dependent tasks after their
predecessor result is joined. Do not call a specialist's tools from the supervisor.

- [ ] **Step 5: Join results and hand back to the main workflow.**

`join_specialist_results` stores all typed results and memory updates. `compose_reply` combines
customer-safe messages in task order, includes RAG citations, and reports when an order ID needs
correction or was not found. `prepare_escalation` creates a main-workflow review request only for a
valid review packet or other existing review-ready action. For `ORDER_NOT_FOUND`, hand control back
with a customer response and no case. `finalise` persists outcome, traces, and any safe memory
update.

- [ ] **Step 6: Wire explicit dependencies in bootstrap and service code.**

Construct the classifier, specialists, memory repository, and graph in `bootstrap.py`; pass them
through `ClientSupportService` rather than using module globals. Preserve the existing RAG index,
SQLite connection, idempotency, and review dispatch wiring. Add operational events for classifier
failure, specialist terminal status, tool failures, review creation, and main-workflow handoff.

- [ ] **Step 7: Run focused tests and commit.**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/workflow/test_supervisor_graph.py tests/unit/services/test_client_support.py tests/unit/workflow/test_graph.py -q`

Expected: PASS. Commit:

```bash
git add src/pwc_support/workflow/graph.py src/pwc_support/domain/state.py src/pwc_support/services/client_support.py src/pwc_support/bootstrap.py tests/unit/workflow/test_supervisor_graph.py tests/unit/services/test_client_support.py
git commit -m "feat: route support tasks through specialist graphs"
```

---

### Task 8: Require human approval for cancellation execution

**Files:**
- Modify: `src/pwc_support/workflow/retail_actions.py:1-180`
- Modify: `src/pwc_support/services/review.py:1-220`
- Modify: `src/pwc_support/domain/models.py:92-106`
- Modify: `app.py:250-420`
- Test: `tests/unit/services/test_cancellation_review.py`

**Interfaces:**
- Produces reviewer decisions `approve_cancellation` and `reject_cancellation`.
- Extends `RetailApprovalService.apply` to support `retail_cancellation` through
  `OrderRepository.cancel(...)`.
- Produces idempotent cancellation execution keyed by `review_id` and expected order version.

- [ ] **Step 1: Write failing review and mutation-boundary tests.**

```python
def test_pending_cancellation_does_not_mutate_order() -> None:
    review = create_cancellation_review()
    assert review.order_status == "processing"
    assert repository.investigate("ORD-1001", "CUS-1001").status == "processing"


def test_approval_cancels_once_and_replay_is_idempotent() -> None:
    review = create_cancellation_review()
    first = decide_review(review.id, "approve_cancellation")
    replay = decide_review(review.id, "approve_cancellation")
    assert first.execution_status == "cancelled"
    assert replay.execution_status == "cancelled"
    assert repository.investigate("ORD-1001", "CUS-1001").status == "cancelled"
    assert repository.count_cancellation_actions(review.id) == 1


def test_rejection_leaves_order_unchanged() -> None:
    review = create_cancellation_review()
    decide_review(review.id, "reject_cancellation")
    assert repository.investigate("ORD-1001", "CUS-1001").status == "processing"
```

Define `create_cancellation_review()` and `decide_review()` test helpers around a seeded database,
review repository, and `ReviewService`; use explicit fixtures so the tests inspect persisted state,
not only returned objects.

- [ ] **Step 2: Run the focused tests and verify they fail.**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/services/test_cancellation_review.py -q`

Expected: FAIL because cancellation is not an approved action type and the service cannot execute it.

- [ ] **Step 3: Implement idempotent reviewer-approved execution.**

Validate that the review packet contains a verified order, customer scope, policy findings, and the
version observed during investigation. Add `retail_cancellation` handling to
`RetailApprovalService.apply`; call `OrderRepository.cancel` only for an approval decision. A
rejection records the decision and audit event without touching the order. Replays with the same
review ID must return the existing execution result, while a changed order version must surface a
reviewable conflict rather than canceling a different state.

- [ ] **Step 4: Preserve action-before-delivery ordering.**

For retail cancellation, execute the idempotent retail action before marking the review complete and
before dispatching the customer outbox message. If the process stops after retail mutation but before
review persistence, replaying the same pending review must reuse the cancellation action and then
finish review/outbox persistence. Do not send a success message for a rejected or failed action.

- [ ] **Step 5: Update reviewer UI and run checks.**

Display the verified order summary, payment/refund implication, fulfilment/shipping state, policy
findings, missing/conflicting facts, recommended action, and provenance. Add cancellation approve/
reject controls while keeping raw database values and private reasoning out of the customer-facing
surface. Run focused tests and Ruff, then commit:

```bash
git add src/pwc_support/workflow/retail_actions.py src/pwc_support/services/review.py src/pwc_support/domain/models.py app.py tests/unit/services/test_cancellation_review.py
git commit -m "feat: require review approval for cancellation"
```

---

### Task 9: Replace the static retail UI with conversational client support

**Files:**
- Modify: `app.py:145-255`
- Modify: `tests/ui/test_retail_contracts.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing UI contract tests.**

```python
def test_client_surface_uses_one_chat_input() -> None:
    source = Path("app.py").read_text()
    assert "st.chat_input" in source
    assert "render_retail_workspace" not in source
    assert "Order ID" not in source
    assert "Item ID" not in source
    assert "Reason" not in source


def test_client_examples_are_natural_language() -> None:
    source = Path("app.py").read_text()
    assert "Can you tell me if my order has shipped?" in source
    assert "What products are on offer?" in source
```

- [ ] **Step 2: Run the UI tests to verify the static contract fails.**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ui/test_retail_contracts.py -q`

Expected: FAIL because the current client tab still renders static product/order/return forms.

- [ ] **Step 3: Render only the conversational client surface.**

Remove `render_retail_workspace()` and its product/order/return forms. Keep one chat history and one
`st.chat_input` that submits every natural-language message to `ClientSupportService`, including
product questions, recommendations, order status, cancellation, return/refund, and policy queries.
Show plain suggested examples as chat starters, not controls that encode a task or collect fields.
Keep the simulated-email channel and human-review/operator surface available as separate internal
demo areas, but do not make client users select a static retail workflow.

- [ ] **Step 4: Render safe agentic trace and review status.**

Show customer-safe specialist messages, citations for RAG answers, and a compact status such as
"checking order details" or "sent for human review". Expose detailed node/tool timing only in the
internal trace view. Do not display raw SQL, policy rule text, hidden prompts, or private reasoning.

- [ ] **Step 5: Update documentation and commit.**

Document the supervisor and specialist topology, compound-intent behavior, order-ID correction flow,
durable memory scope, reviewer approval boundary, and conversational examples. Run the UI tests and
commit:

```bash
git add app.py tests/ui/test_retail_contracts.py README.md
git commit -m "feat: make client support conversational"
```

---

### Task 10: Verify the complete agentic journey and record evidence

**Files:**
- Create: `tests/integration/test_agentic_customer_journey.py`
- Modify: `scripts/run_evaluation.py`
- Modify: `scripts/run_load_test.py`
- Modify: `README.md`

- [ ] **Step 1: Add the real-runtime integration journey.**

Use a seeded SQLite database and the configured local model or deterministic test doubles at the
collaborator boundary. Exercise the service entry point, not only individual nodes:

1. Submit `I want to cancel my order` and assert a clarification asks for an order number.
2. Submit `ORD-9999` in the same conversation and assert a correction request, one lookup attempt,
   and durable active OrderAgent memory.
3. Submit `ORD-8888` and assert a customer-safe no-information response, terminal handoff to the
   main workflow, and no review request or cancellation mutation.
4. Start a new conversation with `Please cancel order ORD-1001`, assert verified order/payment/
   fulfilment facts and a pending human-review request, and assert the order remains unchanged.
5. Approve the review, assert one cancellation action and one customer notification; replay the
   approval and assert no duplicate mutation or notification.
6. Submit a compound question about an order and a policy, assert both specialist results join and
   the policy answer retains RAG citations.

Define a `seeded_service` fixture that constructs the production bootstrap with test configuration,
and use explicit assertions against database rows, review rows, outbox rows, and operational events.

- [ ] **Step 2: Make evaluation and load paths use the real topology.**

Update evaluation cases to include product offer, order status, missing/corrected order ID, second
lookup miss, cancellation review, rejection, approval replay, and compound order-plus-RAG requests.
Ensure evaluation and load scripts call the same bootstrap/service path as the client. Do not report
agentic routing evidence from a graph built without classifier, specialist, repository, or memory
collaborators. Record latency, task count, tool calls, terminal status, review creation, and RAG
citation presence.

- [ ] **Step 3: Run browser and runtime verification.**

Start the application with the supported command, use the client chat for each natural-language
journey, and inspect the internal trace and human-review views. Confirm the static fields are absent,
the second lookup miss hands back to the main workflow, and approval is the only cancellation path.
Capture the observed result and any environment limitation in the README evidence section.

- [ ] **Step 4: Run all final quality gates.**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
ruff check src tests scripts app.py streamlit_app.py
mypy --strict src tests
git diff --check
```

Resolve failures in the scoped feature before claiming completion. Verify that existing RAG,
simulated-email, review, idempotency, and non-retail tests remain green.

- [ ] **Step 5: Commit the evidence-backed verification.**

```bash
git add tests/integration/test_agentic_customer_journey.py scripts/run_evaluation.py scripts/run_load_test.py README.md
git commit -m "test: verify agentic customer support journey"
```

---

## Execution order and review gates

Execute Tasks 1 through 10 in order. Each task must leave its focused tests passing and be committed
before the next task starts. Keep the old static UI until the conversational path is exercised by the
supervisor tests, then remove it in Task 9. Do not run a live evaluation claim until the fake-agent
tests prove task IDs, reducers, memory transitions, and mutation boundaries. Do not claim cancellation
support until the evidence shows: pending review leaves the order unchanged, approval mutates once,
rejection leaves it unchanged, and duplicate approval is idempotent. The final handoff must identify
the commits, test commands, and any environment-dependent checks that could not run.

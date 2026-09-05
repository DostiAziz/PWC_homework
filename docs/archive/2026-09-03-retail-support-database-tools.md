# Retail Support Database Tools and Meaningful Human Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Evolve the current PwC-style RAG prototype into a retail support assistant that answers product questions from SQLite tools and uses asynchronous human review for refund, return, fraud, and policy exceptions.

**Architecture:** Keep LangGraph as the orchestrator and keep RAG for unstructured policy/manual content. Add typed custom LangChain `@tool` wrappers around application services, with read-only product/order queries separated from side-effecting return/refund commands. Deterministic policy code decides whether an action is safe; human review approves exceptional financial or irreversible actions asynchronously.

**Tech Stack:** Python 3.12, LangGraph, LangChain `@tool`, Pydantic v2, SQLite, Streamlit, existing Chroma/Ollama RAG, existing durable case/review/outbox repositories.

**Spec:** `docs/superpowers/specs/2026-09-03-asynchronous-human-review-design.md`, adapted from PwC public-information support to synthetic retail data.

## Global Constraints

- Use synthetic retail data only; no real payment credentials or customer PII.
- Use custom decorated tools for the bounded business operations; do not expose arbitrary model-generated SQL to production writes.
- Product, inventory, order, return, and offer facts must come from SQLite tool results, not generated text.
- Read-only lookups may run automatically; refund, cancellation, goodwill, and policy exceptions require deterministic authorization and often human review.
- All side effects use idempotency keys, optimistic version checks, audit events, and the existing transactional outbox.
- Review decisions never resume a live graph interrupt; they update the durable case and dispatch an approved response asynchronously.
- Run focused tests, full `pytest`, Ruff, and mypy before each task commit.

---

### Task 1: Retail domain contracts and configuration

**Files:**
- Modify: `src/pwc_support/domain/models.py`
- Modify: `src/pwc_support/domain/state.py`
- Modify: `src/pwc_support/config.py`
- Create: `tests/unit/domain/test_retail_models.py`

**Interfaces:**
- `ProductSummary(product_id, name, category, price, currency, stock, attributes, active_offer)`
- `OrderSummary(order_id, customer_id, status, total, currency, items, delivered_at)`
- `ReturnEligibility(eligible, reason, deadline, refund_amount, risk_flags)`
- `ReturnRequest(return_id, order_id, item_id, reason, status, idempotency_key)`
- `RetailIntent` values `PRODUCT_SEARCH`, `PRODUCT_RECOMMENDATION`, `ORDER_STATUS`, `RETURN_REQUEST`, `REFUND_REQUEST`, `GENERAL_POLICY`
- `RetailActionRisk` values `NONE`, `FINANCIAL`, `IRREVERSIBLE`, `FRAUD_SUSPECTED`, `POLICY_EXCEPTION`

- [ ] Add Pydantic validation for currency, positive prices, bounded IDs, and enum values.
- [ ] Add state fields for `retail_intent`, `tool_results`, `action_proposal`, `action_risk`, and `return_id`.
- [ ] Add settings for `retail_db_path`, `refund_auto_approval_limit`, and `return_window_days`.
- [ ] Add tests for valid product/order results and rejection of negative refund amounts or unknown action types.
- [ ] Run `pytest tests/unit/domain/test_retail_models.py -q`, Ruff, and mypy, then commit `feat: add retail support contracts`.

### Task 2: Synthetic retail SQLite schema and seed data

**Files:**
- Modify: `src/pwc_support/storage/database.py`
- Create: `src/pwc_support/storage/retail_schema.sql`
- Create: `scripts/seed_retail_data.py`
- Create: `tests/unit/storage/test_retail_schema.py`

**Tables:** `products`, `inventory`, `offers`, `customers`, `orders`, `order_items`, `return_requests`, `refund_requests`, `retail_audit_events`.

- [ ] Add additive initialization for the tables and indexes on product search fields, order/customer lookup, and pending actions.
- [ ] Keep payment data tokenized as `payment_reference`, never card numbers.
- [ ] Seed at least ten products, multiple stock locations, active/expired offers, delivered and undelivered orders, eligible and late returns, and one high-value order.
- [ ] Test foreign keys, deterministic seed counts, return-window fixtures, and idempotency uniqueness.
- [ ] Run schema tests and commit `feat: add synthetic retail operations schema`.

### Task 3: Typed database repositories

**Files:**
- Create: `src/pwc_support/storage/retail_repositories.py`
- Create: `tests/unit/storage/test_retail_repositories.py`

**Interfaces:**
- `ProductRepository.search(query, category=None, max_price=None, attributes=()) -> tuple[ProductSummary, ...]`
- `ProductRepository.get(product_id) -> ProductSummary | None`
- `ProductRepository.inventory(product_id, location=None) -> int`
- `ProductRepository.offer(product_id) -> dict[str, object] | None`
- `OrderRepository.lookup(order_id, customer_id) -> OrderSummary | None`
- `ReturnRepository.evaluate(order_id, item_id, reason, now) -> ReturnEligibility`
- `ReturnRepository.create(request) -> ReturnRequest`
- `RefundRepository.propose(return_id) -> dict[str, object]`

- [ ] Use parameterized SQL only, bounded result sizes, and requester scoping on orders.
- [ ] Make return creation idempotent by `(order_id, item_id, idempotency_key)`.
- [ ] Use optimistic locking for return/refund status changes.
- [ ] Test product filters, order access denial, eligibility boundaries, duplicate return replay, and conflicting updates.
- [ ] Run focused repository tests and commit `feat: add retail repositories`.

### Task 4: LangChain decorated lookup tools

**Files:**
- Modify: `src/pwc_support/workflow/tools.py`
- Create: `src/pwc_support/workflow/retail_tools.py`
- Create: `tests/unit/workflow/test_retail_tools.py`

**Interfaces:**
- `@tool search_products(query: str, category: str | None = None, max_price: float | None = None) -> dict`
- `@tool get_product(product_id: str) -> dict`
- `@tool check_inventory(product_id: str, location: str | None = None) -> dict`
- `@tool get_active_offer(product_id: str) -> dict`
- `@tool lookup_order(order_id: str, customer_id: str) -> dict`

- [ ] Return structured dictionaries containing source table, timestamp, and bounded result rows.
- [ ] Expose only read-only tools to automatic product/order routes.
- [ ] Validate IDs and enforce customer scoping before querying.
- [ ] Test tool schemas, empty results, access denial, and SQL-injection-shaped input.
- [ ] Run focused tool tests and commit `feat: add decorated retail lookup tools`.

### Task 5: Retail intent routing and product-answer graph path

**Files:**
- Modify: `src/pwc_support/workflow/graph.py`
- Modify: `src/pwc_support/workflow/policy.py`
- Create: `tests/unit/workflow/test_retail_graph.py`

- [ ] Add deterministic intent rules for product search, order status, return, refund, and replacement requests.
- [ ] Use the LLM only for ambiguous intent/entity extraction; it may not authorize a refund or override policy.
- [ ] Route product search/recommendation through lookup tools and render exact product, price, stock, and offer fields.
- [ ] Route order status through scoped order lookup.
- [ ] Use RAG only for policy/manual explanations, combining tool facts with citations.
- [ ] Test that product questions make no case, return exact tool results, and never invent stock or price.
- [ ] Commit `feat: route retail lookup questions through tools`.

### Task 6: Return, refund, and exception workflow

**Files:**
- Create: `src/pwc_support/workflow/retail_actions.py`
- Modify: `src/pwc_support/workflow/graph.py`
- Modify: `src/pwc_support/storage/repositories.py`
- Create: `tests/unit/workflow/test_retail_actions.py`

**Interfaces:**
- `evaluate_return(order_id, item_id, reason, customer_id) -> ReturnEligibility`
- `request_return(..., idempotency_key) -> ReturnRequest | EscalationIntent`
- `propose_refund(return_id) -> ActionProposal`
- `approve_refund(action_id, reviewer_id, expected_version) -> ReviewDecisionResult`

- [ ] Automatically accept only standard low-risk returns within the configured window and below the configured amount.
- [ ] Create a durable case for late, damaged, disputed, high-value, duplicate, or fraud-flagged requests.
- [ ] Return a case or return ID immediately without pretending the refund has completed.
- [ ] Never expose payment references or internal fraud scores to the customer.
- [ ] Test standard return, late return, high-value refund, duplicate request, stale decision, and rejected refund.
- [ ] Commit `feat: add controlled return and refund workflow`.

### Task 7: Asynchronous reviewer and outbox delivery integration

**Files:**
- Modify: `src/pwc_support/services/review.py`
- Modify: `src/pwc_support/services/client_support.py`
- Modify: `src/pwc_support/app.py` or `app.py`
- Create: `tests/unit/services/test_retail_review_delivery.py`

- [ ] Add reviewer actions `approve_refund`, `reject_refund`, `approve_return`, `request_information`, and `offer_replacement`.
- [ ] Move cases through `pending_review -> delivery_pending -> resolved` only after mailbox delivery succeeds.
- [ ] Invoke `OutboxDispatcher` after commit and expose explicit retry for failed delivery.
- [ ] Display case ID, action proposal, amount, policy result, evidence, and response version; keep customer response blank until authored.
- [ ] Show approved responses in the simulated mailbox and client conversation after refresh.
- [ ] Test exactly-once decision replay, dispatch success/failure/retry, and customer-visible reviewed email.
- [ ] Commit `feat: deliver reviewed retail actions asynchronously`.

### Task 8: Retail UI and end-to-end verification

**Files:**
- Modify: `app.py`
- Modify: `README.md`
- Create: `tests/ui/test_retail_contracts.py`
- Create: `tests/integration/test_retail_workflows.py`

- [ ] Add product search and recommendation examples with structured result cards.
- [ ] Add order lookup and return/refund forms using synthetic customer identity.
- [ ] Add reviewer queue cards with action-specific controls and delivery receipts.
- [ ] Hide raw SQL, internal policy rules, fraud flags, and model rationale from customers.
- [ ] Browser-test: product search, product replacement, standard return, exceptional refund, reviewer approval, and mailbox delivery.
- [ ] Run full `pytest`, Ruff, mypy, and a seeded SQLite smoke test.
- [ ] Commit `feat: expose retail support workflows in demo UI`.

## LangChain integration decision

LangChain provides `SQLDatabaseToolkit` and `create_sql_agent`, but the official documentation warns that generic SQL agents can generate expensive or dangerous queries and should use narrowly scoped permissions and application validation. The LangGraph SQL-agent documentation also presents `@tool` wrappers as minimal demonstration tools and explicitly recommends application-specific safeguards. Therefore this plan uses LangChain's `@tool` decorator for typed, bounded business operations and retains direct repository calls for side effects; it does not give the model arbitrary SQL write access.

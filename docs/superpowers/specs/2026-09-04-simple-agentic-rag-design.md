# Simple Agentic RAG Customer Support Design

Date: 2026-09-04
Status: Ready for user review

## Purpose

Build an e-commerce customer-support chatbot that is simple enough to understand and explain,
while meeting the assignment requirements and retaining the intended user-facing behavior:

- answer policy and FAQ questions with cited RAG evidence;
- answer product and offer questions from current database facts;
- show a customer's order status;
- cancel an eligible order after explicit customer confirmation;
- decompose compound questions and combine the results into one reply.

The assignment PDF is a requirements source, not an implementation recipe. It requires an
agentic LangGraph workflow, at least five main-workflow nodes, conditional routing, independent
subtask execution, graph state, at least two tools, a separate modular RAG subgraph, Streamlit,
Docker, functional evaluation, and a small load test. It does not require a multi-agent system.

## Simplicity rules

Every implementation choice must pass these checks:

1. It directly supports a requirement or one of the five user behaviors above.
2. It has one clear responsibility and a testable input/output contract.
3. It does not introduce an abstraction for a single use.
4. It does not preserve an obsolete internal API solely for compatibility.
5. It keeps safety checks around customer data and order mutation.

Internal node names, old constructor signatures, dummy agent classes, and obsolete tests are not
product behavior. They may be replaced. Existing uncommitted work in the source checkout is not
part of this branch and must remain untouched.

## Scope

### Included

- One Streamlit chat interface.
- One LangGraph supervisor workflow.
- One independently callable RAG subgraph.
- Ollama for local generation and embeddings.
- SQLite for products, offers, orders, and cancellation updates.
- Chroma and the existing evidence-selection behavior for cited RAG answers.
- Typed planner output and typed graph boundaries.
- Customer-scoped order reads.
- Confirmation-based, transactional, idempotent cancellation.
- Evaluation, load testing, Docker, and explanatory README documentation.

### Excluded

- Separate ProductAgent, OrderAgent, ReturnRefundAgent, and RagAgent subgraphs.
- Return and refund workflows.
- Human-review queues and reviewer UI.
- Simulated email, mailbox, case, and outbox infrastructure.
- Arbitrary model-selected tools or generated SQL.
- A general dependency graph between planned tasks.
- Production authentication, payments, fulfilment, and distributed processing.

These exclusions remove unfinished or unnecessary machinery. They do not remove an approved
chatbot behavior.

## Data and model choices

Use a small, versioned retail-support corpus containing shipping, cancellation, warranty, and FAQ
documents. Keep a manifest with source IDs and checksums so ingestion is reproducible. Product,
offer, inventory, and order records remain in SQLite and are never treated as RAG documents.

Use the existing local `gpt-oss:20b` generation model and `nomic-embed-text` embedding model through
Ollama. The generation model fits the available 24 GB machine and follows grounded-answer and
structured-output instructions better than the previously measured smaller alternative, at the cost
of higher latency and serialized generation. This trade-off must be reported from the real evaluation
and load runs. The design does not add a model-provider abstraction because only Ollama is required.

## System model

The system has three kinds of work:

| Work | Source of truth | Model responsibility |
|---|---|---|
| Policy and FAQ answers | RAG documents | Write only from selected evidence |
| Products, prices, offers, order status | SQLite | Classify the request, never invent facts |
| Order cancellation | Deterministic application command | Extract intent, never authorize mutation |

The LLM decides where a request belongs and writes grounded prose. Application code controls data
access, validation, customer scope, eligibility, confirmation, and mutation.

## Main LangGraph workflow

```text
                         +-> rag_task -> RAG subgraph --------+
                         |                                     |
intake -> plan_tasks ----+-> catalogue_task -------------------+-> join_results -> respond
                         |                                     |
                         +-> order_task ------------------------+
```

The main graph contains seven nodes:

1. `intake`: normalize the message and bind the session's customer identity.
2. `plan_tasks`: produce one or more closed task types.
3. `rag_task`: invoke the RAG subgraph for policy or FAQ questions.
4. `catalogue_task`: execute product and offer reads.
5. `order_task`: execute order status, cancellation preview, or confirmed cancellation.
6. `join_results`: collect independently produced results in deterministic task order.
7. `respond`: create one customer-facing answer and expose citations and a concise trace.

The RAG invocation does not need to count toward the assignment's main-node requirement. The
remaining six nodes still satisfy the stricter "at least five" wording in the PDF.

`plan_tasks` uses LangGraph `Send` to dispatch independent tasks. A message such as "What jackets
are on offer, and cancel order ORD-123" becomes a catalogue task and an order task. Their results
are joined into one reply. There is no task-dependency DAG because the approved use cases do not
need one.

## Graph state

Use one small `TypedDict` for internal graph state. Use Pydantic models only at untrusted or
persistent boundaries.

```python
class SupportState(TypedDict, total=False):
    message: str
    customer_id: str
    pending_cancellation: CancellationPreview | None
    tasks: list[Task]
    results: Annotated[list[TaskResult], operator.add]
    response: str
    citations: list[Citation]
    events: Annotated[list[Event], operator.add]
```

Only fields consumed by a later node belong in state. Private model reasoning, raw SQL, payment
data, and duplicate projections of the same value do not.

## Planner contract

The Ollama planner returns a closed Pydantic schema:

```python
TaskKind = Literal["knowledge", "catalogue", "order"]

class Task(BaseModel):
    id: str
    kind: TaskKind
    request: str
    product_query: str | None = None
    order_id: str | None = None
    action: Literal["lookup", "cancel"] | None = None
```

Rules:

- At most four tasks per message.
- Task kinds and actions are closed enums.
- The model cannot provide `customer_id`.
- Greetings and confirmation replies use deterministic fast paths.
- Invalid model output produces an explicit unavailable or clarification response.

## Business tools

Tools are ordinary typed Python functions or small service methods. A framework-specific tool
wrapper is unnecessary unless LangGraph directly requires it.

- `search_products(query)`: return matching active products and current prices.
- `list_offers(query=None)`: return active offers and computed effective prices.
- `get_order(customer_id, order_id)`: return only an order owned by that customer.
- `preview_cancellation(customer_id, order_id)`: validate ownership and cancellation eligibility
  without mutation.
- `cancel_order(customer_id, order_id, confirmation_token)`: apply one eligible cancellation in a
  transaction and return the existing result on replay.

Product, offer, and order facts come from SQLite. They are not embedded in the RAG corpus because
they are structured and can change.

## Cancellation conversation

1. The customer asks to cancel an order.
2. If the order ID is missing, the chatbot asks for it.
3. `preview_cancellation` checks customer scope, status, and eligibility.
4. The chatbot states the exact order and asks for explicit confirmation.
5. The pending preview is stored in session state with an opaque confirmation token.
6. A clear confirmation invokes `cancel_order` with that token.
7. The command validates the preview again and updates the order transactionally.
8. Repeating the confirmation returns the same result without a second mutation.

The LLM never decides that confirmation occurred and never performs the database update directly.
When a cancellation is pending, a small deterministic yes/no parser handles the next message.

## RAG subgraph

Retain the existing four-stage responsibility split:

1. `prepare_query`: normalize the knowledge question.
2. `retrieve`: retrieve candidates from the indexed corpus.
3. `select_evidence`: apply the existing relevance threshold and evidence limits.
4. `answer_with_citations`: generate only from selected evidence and validate citation markers.

The first simplification phase does not replace the existing retrieval algorithm. Retrieval can be
simplified later only if the frozen evaluation set shows no material regression in grounded-answer
quality or latency.

## Ollama boundary

Ollama integration should be a thin boundary, not an application framework. It needs only:

- one configured `ollama.Client`;
- text generation;
- schema-constrained generation with Pydantic validation;
- batched embeddings;
- transport timeout normalization;
- one shared concurrency limit for the local model.

`text` and `structured` must share one private chat call instead of duplicating request assembly and
timeout handling. Client construction belongs in bootstrap code. Ingestion-specific prompt building
belongs in the ingestion module, not in the Ollama adapter. No agent-specific Ollama planner classes
are needed.

The target is approximately 80-110 readable lines, but responsibility and tests take priority over
an arbitrary line count.

## UI and observability

Streamlit presents one chat conversation. It may show a compact expandable trace containing:

- planned task types;
- nodes visited;
- tool names and durations;
- cited sources;
- cancellation state.

It must not expose prompts, chain-of-thought, raw model output, SQL, or hidden customer data. There
is no second reviewer or email workspace.

## Runtime and persistence

- Use one SQLite retail database for products, offers, orders, and cancellation tokens.
- Keep Chroma and the lexical index required by the current RAG implementation.
- Keep configuration in one Pydantic settings model, with environment variables only for values
  that differ between local and Docker runs.
- Keep one composition root that constructs repositories, tools, Ollama, RAG, and the graph.
- Docker Compose may run Streamlit and Chroma while Ollama remains on the host.

## Error behavior

- Ollama unavailable: return an explicit local-model unavailable response.
- Empty or unavailable knowledge index: return an explicit knowledge-base unavailable response.
- Insufficient evidence: abstain rather than generate an unsupported answer.
- Unknown product or order: return a scoped not-found response without leaking other customers'
  records.
- Ineligible cancellation: explain the deterministic reason and do not create a pending action.
- Database failure during cancellation: roll back and return an error without claiming success.

## Migration strategy

The current branch has 148 passing and 20 failing tests before this redesign. The failures show that
the old workflow API and the incomplete specialist implementation are already inconsistent. They
are not a green compatibility target.

Implementation must proceed vertically:

1. Add characterization tests for the approved user journeys.
2. Implement the minimal graph and tools until those tests pass.
3. Point Streamlit and runtime bootstrap to the minimal graph.
4. Verify the live journeys and real Ollama/RAG path.
5. Remove superseded specialist, review, email, and case code only after reference searches confirm
   there are no remaining runtime imports.
6. Replace obsolete tests with tests of the approved public behavior.

Do not build compatibility adapters for removed `build_graph` signatures. They would preserve
accidental complexity rather than product behavior.

## Verification

### Characterization tests

- A policy question produces a grounded answer with valid citations.
- A product query returns only database products and prices.
- An offer query returns active offers and effective prices.
- An order lookup cannot expose another customer's order.
- A cancellation request does not mutate before confirmation.
- A confirmed eligible cancellation mutates exactly once.
- An ineligible cancellation never mutates.
- A compound knowledge and commerce message produces independent tasks and one joined reply.
- Model, retrieval, and database failures produce explicit safe outcomes.
- The compiled main graph exposes at least five non-RAG nodes.

### Quality gates

- Full pytest suite passes with no obsolete tests retained.
- Ruff, formatting, strict mypy, compileall, and lock-file checks pass.
- Docker image and Compose configuration validate.
- Streamlit live checks cover knowledge, products, offers, order status, cancellation preview,
  confirmation, and one compound request.
- The 10-20 case functional evaluation runs against the real graph and records groundedness,
  routing correctness, task completion, and latency.
- The 50-200 query load run exercises the real configured runtime and reports p50, p95, p99,
  throughput, failures, the measured bottleneck, and one or two justified optimizations.

## Explanation for reviewers

The architecture can be explained in one sentence:

> The planner splits a customer message into knowledge, catalogue, or order tasks; LangGraph runs
> independent tasks, the RAG subgraph answers document questions with citations, deterministic
> tools handle live business data and confirmed cancellation, and the join node returns one reply.

That sentence should remain true when reading the code. If explaining a component requires a new
architecture concept, the component must justify itself against the requirements before it is added.

# Customer Support Assistant Architecture

## 1. Retail Problem and Domain Context

Retail customer support involves handling high volumes of recurring customer inquiries across two fundamentally different types of data:
1. **Dynamic, Structured Retail Operations**: Product catalogues, inventory availability, promotional offers, and customer-specific orders.
2. **Static / Semi-Structured Policies**: Return terms, warranty conditions, cancellation rules, and shipping timelines.

Answering retail inquiries requires strict accuracy:
- Hallucinating prices, inventory, or order statuses leads directly to financial and customer trust damage.
- Mutating state (e.g., cancelling an order) without explicit human confirmation or customer authorization risks unauthorized modifications and race conditions.
- Misquoting policy terms or inventing terms causes regulatory and warranty compliance issues.

## 2. Architecture & Design Rationale

### Why Structured Tools + Policy RAG?

A dual-store architecture cleanly partitions responsibilities:
- **Relational SQLite Database (`retail-support-v1.sqlite3`)**: Provides ACID guarantees, relational integrity, and strict customer authorization filters (`WHERE customer_id = ?`) for orders, products, and discounts.
- **Hybrid RAG Knowledge Base (ChromaDB + SQLite BM25)**: Delivers semantic and lexical retrieval with Reciprocal Rank Fusion (RRF) over Markdown policy documents, enforcing verified citation markers `[S#]` linked to source evidence.

### Why a 5-Node Single-Agent Graph is Sufficient

Rather than building an over-engineered multi-agent hierarchy with supervisor orchestrators and specialist sub-agents that introduce fragility and latency, the system uses a single unified LangGraph tool-calling loop:

```text
Streamlit chat -> AgentService (thread_id, customer_id)
                     |
START -> intake -> agent <----------------------------.
                     |                                |
                     |-- tool_calls (non-cancellation) -> tools ->'
                     |-- cancel_order ---------------> confirm (interrupt) ->'
                     `-- final answer ---------------> respond -> END
```

### Node Responsibilities

1. **`intake`**:
   - Resets per-turn metadata (citations, retrieved evidence, execution logs).
   - Injects base system instructions and customer identity context into the thread history.
2. **`agent`**:
   - The primary reasoning node running `gpt-oss:20b` via Ollama.
   - Evaluates conversation history and available tool definitions to either emit tool calls or generate the final user response.
3. **`tools`**:
   - Dispatches read-only tool calls emitted by the agent:
     - `search_products(query)`: Searches catalogue by name, category, and description.
     - `list_offers(category?)`: Lists active promotional discounts and terms.
     - `get_order_status(order_id)`: Retrieves customer-scoped order details.
     - `search_policies(question)`: Invokes the nested 4-node RAG subgraph.
   - Preserves compound/mixed tool call responses without dropping messages.
4. **`confirm`**:
   - Safety boundary for state mutation (`cancel_order`).
   - Validates order existence, checks customer ownership, and confirms that the order is in `processing` status.
   - Generates an immutable `CancellationPreview` and pauses execution via LangGraph `interrupt()`.
   - On resume with customer approval (`"yes"`), executes `OrderRepository.cancel` using an atomic `BEGIN IMMEDIATE` transaction and optimistic version lock. If rejected or disputed, reports refusal back to the agent without mutating state.
5. **`respond`**:
   - Validates that citation markers in the agent's final text correspond to genuine retrieved citations.
   - Emits the validated `ChatReply` with structured status, timing spans, and citations.

### Nested RAG Subgraph (4 nodes)

The `search_policies` tool delegates to a dedicated 4-node subgraph:
1. **`prepare_query`**: Sanitizes punctuation and prepares query strings.
2. **`retrieve_candidates`**: Performs hybrid vector search (ChromaDB with `sentence-transformers/all-MiniLM-L6-v2`) and lexical search (SQLite BM25), combined using Reciprocal Rank Fusion (RRF).
3. **`select_evidence`**: Filters candidates against a minimum cosine similarity threshold (0.45), respects token budgets, and formats citation references.
4. **`answer_with_citations`**: Formulates the policy synthesis prompt requiring strict `[S#]` citations, validates claim markers, and falls back to safe refusal if evidence is insufficient or contradictory.

## 3. Local Model & Resource Strategy

- **Generation Model**: `gpt-oss:20b` running via local Ollama. It provides tool-calling and structured instruction-following capabilities on local hardware.
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors). It runs via local HuggingFace Embeddings, ensuring deterministic ingestion and query representations.
- **Concurrency Control**: `LimitedChatModel` enforces a concurrency limiter (`max_parallel_generations`) with an acquisition timeout to prevent GPU memory saturation on local machines.
- **Span Profiling**: Request-scoped execution spans (`generation.queue`, `generation.invoke`, `retrieval`, `workflow`) capture true wall-clock bottlenecks without fabricated metrics.

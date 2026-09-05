# Agentic Tool-Calling Redesign — Design Spec

Date: 2026-09-05
Status: Approved for planning
Branch: `claude/system-testing-continuation-57da16`

## Goal

Replace the deterministic planner + graph router with a single **tool-calling
agent** — one LLM "brain" that reads the conversation and autonomously decides
which tool to call. The agent is built as an **explicit custom LangGraph** (not
the prebuilt `create_react_agent`) so it still satisfies every structural
requirement in the project brief: a ≥5-node graph with autonomous conditional
routing, subtask decomposition, state management, and a dedicated modular RAG
subgraph.

The behaviour the user wants: "a model you chat with that also fetches offers,
looks up products, and cancels orders" — with the LLM in charge of all
decisions, not a keyword classifier.

## Locked decisions

1. **Scope:** rip out the planner and the `plan_tasks` / `route_tasks` /
   `join_results` routing; reuse the existing tool bodies, Streamlit UI, SQLite,
   Chroma, corpus, and `OllamaGateway`.
2. **Cancellation safety:** agent proposes, user confirms. The agent may never
   commit a cancellation on its own; a LangGraph `interrupt()` pauses for an
   explicit yes/no.
3. **Grounding:** strict. Policy answers come only from retrieved corpus text
   with `[S1]`-style citations, or the system refuses. No answering policy from
   the model's own knowledge.
4. **Memory:** multi-turn via a LangGraph checkpointer. Use `MemorySaver`
   (in-memory) for now; `SqliteSaver` is a later one-line swap.
5. **Implementation:** custom LangGraph + native Ollama tool-calling. **No new
   dependencies** (`langgraph` and `ollama` are already present).
6. **Customer scoping:** the order tools receive `customer_id` from session
   state, injected by the runtime — the model cannot request another customer's
   data.

## Architecture

### Main graph — 5 nodes

| # | Node | Responsibility | Decisions? |
|---|------|----------------|------------|
| 1 | `intake` | Normalize input, ensure thread state | none |
| 2 | `agent` | LLM sees conversation + tool schemas, chooses tool(s) or final answer | **all** |
| 3 | `tools` | Execute the tool call(s) the agent requested; return results | none |
| 4 | `confirm` | Human-in-the-loop `interrupt()` gate for cancellation commit | none (relays yes/no) |
| 5 | `respond` | Assemble final message + citations for the UI | none |

**Conditional edge after `agent`:** "did the model request a tool? → `tools`,
else → `respond`." This is the brief's "autonomous decision-making (conditional
routing)" — decided by the model, not by hand-written `if kind == ...`.

Only `agent` has intelligence. The other four nodes are mechanical.

### RAG subgraph — 4 nodes (separate; does not count toward the 5)

`prepare_query → retrieve → select_evidence → answer_with_citations`, unchanged
from today, invoked as the `search_policies` tool.

### Tools (the agent's hands)

| Tool | Backed by | Retrieval-based? |
|------|-----------|------------------|
| `search_products(query)` | `CommerceTools` / SQLite | no |
| `list_offers(category?)` | `CommerceTools` / SQLite | no |
| `get_order_status(order_id)` | SQLite, scoped to session `customer_id` | no |
| `cancel_order(order_id)` | SQLite; preview then `confirm`-gated commit | no |
| `search_policies(question)` | 4-node RAG subgraph via `RagAnswerer` | yes |

Satisfies "≥2 tools, ≥1 not purely retrieval-based": four SQLite tools plus one
RAG tool.

## Data flow

### Normal turn
```
submit(thread_id, text)
  intake → agent → [tool requested?] → tools → agent → … → respond → ChatReply
                        │ no
                        └────────────────────────────────▶ respond
```
The agent may loop (call tool, read result, call another, then answer). The
checkpointer persists the message history under `thread_id`, giving multi-turn
memory.

### Cancellation turn
```
"cancel ORD-2001" → agent calls cancel_order(ORD-2001)
                    → confirm node runs interrupt({preview})   ← graph PAUSES
  submit() returns {status: "awaiting_confirmation", preview}
  UI shows the preview.
"yes" → resume(thread_id, "yes") → Command(resume=...)
        → commit via existing OrderRepository.cancel()  → agent: "Order cancelled."
"no"  → resume rejects → agent: "Order was not cancelled."
```
The commit reuses the existing atomic cancellation (`BEGIN IMMEDIATE`, version
check, single audit row). Every safety property verified in prior testing is
preserved.

## Memory

- `MemorySaver` created once and cached via `@st.cache_resource`.
- Each browser session holds a stable `thread_id` in `st.session_state`.
- In-memory: conversation persists for the life of the server process; a restart
  clears it. `SqliteSaver` is a later swap for durability.

## Grounding firewall

`search_policies` returns the **finished, cited answer** from `RagAnswerer`
(which already retrieves, validates citations, and refuses when nothing is
grounded) — not raw passages. The system prompt instructs the agent to relay
policy answers verbatim with their `[S1]` markers and never state policy from its
own knowledge. Citations flow from the tool result to the UI unchanged.

## Error handling

- Tool failure (SQLite / Chroma / Ollama unavailable) → tool returns a safe
  message string; the agent relays it. Same safe wording as today.
- **Max-iterations guard** on the agent loop (default 6 tool rounds); on exceed,
  a safe fallback message.
- Invalid tool arguments → rejected by the tool schema; the agent receives an
  error result and can retry.
- Model unavailable → caught in `AgentService.submit`, safe message returned.

## Components and files

**New / heavily changed**
- `src/pwc_support/workflow/agent_graph.py` — the 5-node graph, conditional
  edge, `interrupt()` confirm, `MemorySaver`.
- `src/pwc_support/workflow/tools.py` — the five tool definitions (schemas +
  dispatch), wrapping existing `CommerceTools` / `RagAnswerer`.
- `src/pwc_support/services/chat.py` — `AgentService` with `submit(thread_id,
  text)` and `resume(thread_id, decision)`; returns `ChatReply` incl.
  `awaiting_confirmation` + `preview`.
- `src/pwc_support/llm/ollama.py` — add `chat_with_tools(messages, tools)`.
- `src/pwc_support/bootstrap.py` — build and cache the new agent graph +
  `MemorySaver` instead of the old `build_graph`.
- `app.py` — thread a `thread_id`; route yes/no to `resume()` when the last turn
  was `awaiting_confirmation`; trace expander shows the agent's tool steps.

**Deleted / replaced**
- `src/pwc_support/workflow/planner.py` — deleted.
- `src/pwc_support/workflow/graph.py` — the old `build_graph` (with
  `plan_tasks` / `route_tasks` / `join_results`) is superseded by
  `agent_graph.py`; remove the old module once bootstrap points at the new one.

**Kept and reused unchanged**
- `CommerceTools` data methods, `OrderRepository`, `ProductRepository`
- `RagAnswerer` and the 4-node RAG subgraph, `store.py`, `ingest.py`, `lexical.py`
- `config.py`, `database.py`, `OllamaGateway.embed/text/structured`
- Streamlit shell, Dockerfile, `compose.yaml`, corpus, seed/ingest scripts

## Testing and evaluation

- **Unit tests:** new tests for each tool, the agent graph (routing loop, max
  iterations), the `interrupt()`/`resume()` cancellation cycle, grounding
  relay, and customer scoping. Remove planner tests. Keep RAG, repository, and
  subgraph tests unchanged.
- **Eval (brief requires 10–20 questions):** re-point the frozen 16-case set
  from exact-string assertions to **intent checks** — correct tool selected,
  citations present for policy answers, cancellation still gated, cross-customer
  blocked, no hallucination on out-of-scope. Better aligned with the rubric's
  "evaluation methodology" criterion.
- **Load test:** unchanged path via `AgentService.submit`.

## Requirements mapping (project brief → design)

| Brief requirement | Satisfied by |
|-------------------|--------------|
| Agentic RAG chatbot, Python, LangGraph | Custom LangGraph agent + RAG subgraph |
| ≥5 nodes | intake, agent, tools, confirm, respond |
| Autonomous decision-making (conditional routing) | Model-driven conditional edge after `agent` |
| Decomposition into subtasks + independent execution | Agent issues multiple tool calls per turn, executed in `tools` |
| State management for intermediate results | Graph state: message history + accumulated tool results (checkpointer) |
| Dedicated modular RAG subgraph (not counted in nodes) | 4-node subgraph via `search_policies` |
| ≥2 tools, ≥1 not purely retrieval-based | 4 SQLite tools + 1 RAG tool |
| Text-based data source | Existing corpus (shipping, warranty, cancellation) |
| Open-source local LLM (or dummy) | `gpt-oss:20b` via Ollama (tool-calling verified) |
| Streamlit UI showing key agent steps + RAG output | Chat UI + trace expander listing tool calls + citations |
| Containerized, Dockerfile mandatory, compose an advantage | Existing Dockerfile + `compose.yaml`, unchanged |
| Functional eval (10–20 questions) | Re-pointed 16-case intent-based eval |
| Load test (50–200 queries), latency, bottleneck, optimization | Existing load harness, re-run on new path |
| README (problem, architecture, results, install/run) | Update README for the new architecture |

## Out of scope (YAGNI)

- Persistent conversation storage (`SqliteSaver`) — deferred; `MemorySaver` now.
- Multiple customers / auth — the demo stays single-customer (`CUS-1001`).
- Streaming token output in the UI.
- Editing tool arguments during the confirmation interrupt (only approve/reject).
- Any new runtime dependency.

## Open follow-ups (post-implementation)

- Decide final eval scoring thresholds once agent phrasing is observed.
- Update README architecture diagram and node descriptions.

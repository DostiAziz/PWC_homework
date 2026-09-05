# Agentic Tool-Calling Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the deterministic planner + graph router with a single tool-calling agent (one LLM "brain") built as an explicit custom LangGraph, so the LLM decides every action while still satisfying the brief's ≥5-node + RAG-subgraph structure.

**Architecture:** A 5-node LangGraph (`intake → agent → {tools|confirm} → agent → respond`) where only the `agent` node thinks — it calls the LLM with tool schemas and the model chooses tools. Non-retrieval tools hit SQLite; the `search_policies` tool invokes the existing 4-node RAG subgraph. Cancellation commits only after a LangGraph `interrupt()` confirmation. Multi-turn memory via an in-memory `MemorySaver` checkpointer.

**Tech Stack:** Python 3.12, LangGraph 1.2.11 (`StateGraph`, `interrupt`, `MemorySaver`), Ollama 0.6.2 native tool-calling (`gpt-oss:20b`), Pydantic 2, SQLite, Chroma, Streamlit. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-05-agentic-tool-calling-redesign-design.md`

## Global Constraints

- No new runtime dependencies — only `langgraph` and `ollama` (already present).
- Agent may never commit a cancellation without an explicit user "yes" via `interrupt()`.
- Policy answers only from `RagAnswerer` output with `[S1]` citations, or refuse — never from the model's own knowledge.
- Order tools receive `customer_id` from the runtime, never from model-supplied arguments (single demo customer `CUS-1001`).
- Memory: `MemorySaver` (in-memory) for now.
- TDD: write the failing test first, watch it fail, implement minimally, watch it pass, commit.
- Run tests with: `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`
  (the venv is the main repo's: `/Users/admin/PycharmProjects/Customer Support/.venv`).

---

## Preliminary: verify clean baseline

The earlier uncommitted worktree edits (planner offers/compound + `awaiting_cancel`) have been **committed and merged into main** (`27f6f56`). The redesign deletes `planner.py`, `graph.py`, and `commerce.py` entirely (Task 7) and fully rewrites `chat.py` (Task 5) and `app.py` (Task 6), so those merged changes do not conflict.

- [ ] **Step 1: Verify working tree is clean**

```bash
git status   # working tree clean except possibly this plan file
git log --oneline -1   # should show 27f6f56 or later
```
Expected: clean working tree on `main` at commit `27f6f56` or later.

---

## Task 1: `OllamaGateway.chat_with_tools`

**Files:**
- Modify: `src/pwc_support/llm/ollama.py`
- Test: `tests/unit/llm/test_ollama.py`

**Interfaces:**
- Consumes: existing `OllamaGateway.__init__(client, generation_model=..., ...)`.
- Produces: `OllamaGateway.chat_with_tools(messages: list[dict], tools: list[dict], *, max_tokens: int = 512, temperature: float = 0.0) -> dict` returning the assistant message dict `{"role": "assistant", "content": str, "tool_calls": list[dict]}` (tool_calls normalized to `[{"name": str, "arguments": dict}]`, empty list when none).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/llm/test_ollama.py (append)
def test_chat_with_tools_normalizes_tool_calls() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.captured: dict = {}

        def chat(self, **kwargs):
            self.captured = kwargs
            return {"message": {"role": "assistant", "content": "",
                    "tool_calls": [{"function": {"name": "get_order_status",
                                                 "arguments": {"order_id": "ORD-1"}}}]}}

    client = FakeClient()
    gw = OllamaGateway(client, generation_model="m", embedding_model="e")
    tools = [{"type": "function", "function": {"name": "get_order_status",
              "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}}}]
    msg = gw.chat_with_tools(messages=[{"role": "user", "content": "where is ORD-1"}], tools=tools)

    assert msg["tool_calls"] == [{"name": "get_order_status", "arguments": {"order_id": "ORD-1"}}]
    assert client.captured["tools"] == tools
    assert client.captured["messages"][0]["content"] == "where is ORD-1"


def test_chat_with_tools_returns_empty_tool_calls_for_plain_answer() -> None:
    class FakeClient:
        def chat(self, **kwargs):
            return {"message": {"role": "assistant", "content": "Hello."}}

    gw = OllamaGateway(FakeClient(), generation_model="m", embedding_model="e")
    msg = gw.chat_with_tools(messages=[{"role": "user", "content": "hi"}], tools=[])
    assert msg["content"] == "Hello."
    assert msg["tool_calls"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/llm/test_ollama.py -q`
Expected: FAIL — `AttributeError: 'OllamaGateway' object has no attribute 'chat_with_tools'`.

- [ ] **Step 3: Implement `chat_with_tools`**

```python
# src/pwc_support/llm/ollama.py  (add method to OllamaGateway)
def chat_with_tools(
    self,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> dict[str, Any]:
    options: dict[str, float | int] = {"temperature": temperature, "num_predict": max_tokens}
    if self.num_ctx is not None:
        options["num_ctx"] = self.num_ctx
    request: dict[str, Any] = {"model": self.generation_model, "messages": messages, "options": options}
    if tools:
        request["tools"] = tools
    try:
        with self.generation_slots:
            response = self.client.chat(**request)
        message = response["message"]
    except (HTTPError, ollama.RequestError, ollama.ResponseError, KeyError, TypeError) as error:
        raise OllamaUnavailable("Ollama tool chat failed") from error
    raw_calls = message.get("tool_calls") or []
    tool_calls = [
        {"name": call["function"]["name"], "arguments": dict(call["function"].get("arguments") or {})}
        for call in raw_calls
    ]
    return {"role": "assistant", "content": message.get("content") or "", "tool_calls": tool_calls}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/llm/test_ollama.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pwc_support/llm/ollama.py tests/unit/llm/test_ollama.py
git commit -m "feat: add native tool-calling to Ollama gateway"
```

---

## Task 2: Non-cancel tools + registry (`tools.py`)

**Files:**
- Create: `src/pwc_support/workflow/tools.py`
- Test: `tests/unit/workflow/test_tools.py`

**Interfaces:**
- Consumes: `ProductRepository.search`, `ProductRepository.list_active_offers`, `OrderRepository.lookup`, `RagAnswerer.answer(RagRequest)`.
- Produces:
  - `TOOL_SCHEMAS: list[dict]` — Ollama tool schemas for `search_products`, `list_offers`, `get_order_status`, `search_policies`, `cancel_order`.
  - `class ToolRegistry` with `__init__(self, products: ProductRepository, orders: OrderRepository, rag: RagAnswerer)` and `run(self, name: str, arguments: dict, *, customer_id: str) -> ToolOutcome`.
  - `@dataclass ToolOutcome: content: str; citations: tuple[Citation, ...] = (); requires_confirmation: bool = False; order_id: str | None = None`.

Note: `run("cancel_order", ...)` returns `ToolOutcome(content="", requires_confirmation=True, order_id=...)` WITHOUT touching the DB — the `confirm` node (Task 4) performs the preview/commit. All other tools return their content directly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/workflow/test_tools.py
from pwc_support.rag.answer import RagAnswerer
from pwc_support.storage.database import Database
from pwc_support.storage.retail_repositories import OrderRepository, ProductRepository
from pwc_support.workflow.tools import TOOL_SCHEMAS, ToolRegistry
from tests.fakes import FakeGenerator, FakeKnowledgeBase


def _registry(retail_db: Database) -> ToolRegistry:
    rag = RagAnswerer(FakeKnowledgeBase(), FakeGenerator())
    return ToolRegistry(ProductRepository(retail_db), OrderRepository(retail_db), rag)


def test_schemas_expose_five_tools() -> None:
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert names == {"search_products", "list_offers", "get_order_status",
                     "search_policies", "cancel_order"}


def test_list_offers_returns_discounted_price(retail_db: Database) -> None:
    out = _registry(retail_db).run("list_offers", {"category": "jackets"}, customer_id="CUS-1001")
    assert "116.99" in out.content
    assert out.requires_confirmation is False


def test_get_order_status_is_customer_scoped(retail_db: Database) -> None:
    reg = _registry(retail_db)
    mine = reg.run("get_order_status", {"order_id": "ORD-5001"}, customer_id="CUS-1001")
    assert "shipped" in mine.content.lower()
    other = reg.run("get_order_status", {"order_id": "ORD-3001"}, customer_id="CUS-1001")
    assert "could not find" in other.content.lower()


def test_search_policies_returns_citations(retail_db: Database) -> None:
    out = _registry(retail_db).run("search_policies", {"question": "how long is shipping?"},
                                   customer_id="CUS-1001")
    assert "[S1]" in out.content
    assert len(out.citations) >= 1


def test_cancel_order_defers_to_confirmation(retail_db: Database) -> None:
    out = _registry(retail_db).run("cancel_order", {"order_id": "ORD-2001"}, customer_id="CUS-1001")
    assert out.requires_confirmation is True
    assert out.order_id == "ORD-2001"


def test_unknown_tool_returns_safe_message(retail_db: Database) -> None:
    out = _registry(retail_db).run("nope", {}, customer_id="CUS-1001")
    assert "unknown" in out.content.lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/workflow/test_tools.py -q`
Expected: FAIL — `ModuleNotFoundError: pwc_support.workflow.tools`.

- [ ] **Step 3: Implement `tools.py`**

```python
# src/pwc_support/workflow/tools.py
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from pwc_support.domain.models import Citation, RagRequest
from pwc_support.rag.answer import RagAnswerer
from pwc_support.storage.retail_repositories import OrderRepository, ProductRepository

TOOL_SCHEMAS: list[dict] = [
    {"type": "function", "function": {
        "name": "search_products",
        "description": "Search the retail catalogue for products by name or category.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "list_offers",
        "description": "List active discount offers, optionally filtered by product category.",
        "parameters": {"type": "object",
                       "properties": {"category": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_order_status",
        "description": "Get the status of the current customer's order by order id.",
        "parameters": {"type": "object",
                       "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "search_policies",
        "description": "Answer a shipping, warranty, or cancellation policy question from the docs.",
        "parameters": {"type": "object",
                       "properties": {"question": {"type": "string"}}, "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "cancel_order",
        "description": "Start cancelling the current customer's order. Requires user confirmation.",
        "parameters": {"type": "object",
                       "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
]


@dataclass
class ToolOutcome:
    content: str
    citations: tuple[Citation, ...] = ()
    requires_confirmation: bool = False
    order_id: str | None = None


class ToolRegistry:
    def __init__(self, products: ProductRepository, orders: OrderRepository, rag: RagAnswerer) -> None:
        self.products = products
        self.orders = orders
        self.rag = rag

    def run(self, name: str, arguments: dict, *, customer_id: str) -> ToolOutcome:
        try:
            handler = getattr(self, f"_{name}", None)
            if handler is None:
                return ToolOutcome(content=f"Unknown tool: {name}.")
            return handler(arguments, customer_id)
        except sqlite3.Error:
            return ToolOutcome(content="That data source is unavailable. Please try again.")

    def _search_products(self, args: dict, customer_id: str) -> ToolOutcome:
        products = self.products.search(str(args.get("query", "")))
        body = "\n".join(
            f"{p.name}: {p.price} {p.currency}, {p.stock} in stock" for p in products
        ) or "No available products matched your request."
        return ToolOutcome(content=body)

    def _list_offers(self, args: dict, customer_id: str) -> ToolOutcome:
        offers = self.products.list_active_offers(query=args.get("category"))
        body = "\n".join(
            f"{o.name} ({o.product_id}): {o.effective_price} {o.currency}, {o.description}"
            for o in offers
        ) or "No active offers matched your request."
        return ToolOutcome(content=body)

    def _get_order_status(self, args: dict, customer_id: str) -> ToolOutcome:
        order = self.orders.lookup(str(args.get("order_id", "")), customer_id)
        if order is None:
            return ToolOutcome(content="I could not find that order for this customer.")
        return ToolOutcome(content=f"Order {order.order_id} is {order.status}.")

    def _search_policies(self, args: dict, customer_id: str) -> ToolOutcome:
        result = self.rag.answer(RagRequest(question=str(args.get("question", ""))))
        if result.status != "answered":
            return ToolOutcome(content=(
                "I could not find grounded policy information for that question. "
                "Our retail support documents do not contain that information."))
        return ToolOutcome(content=result.answer, citations=result.citations)

    def _cancel_order(self, args: dict, customer_id: str) -> ToolOutcome:
        return ToolOutcome(content="", requires_confirmation=True,
                           order_id=str(args.get("order_id", "")))
```

- [ ] **Step 4: Run to verify pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/workflow/test_tools.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pwc_support/workflow/tools.py tests/unit/workflow/test_tools.py
git commit -m "feat: add agent tool registry over existing repositories"
```

---

## Task 3: Cancellation preview builder + commit helpers

**Files:**
- Modify: `src/pwc_support/workflow/tools.py`
- Test: `tests/unit/workflow/test_tools.py`

**Interfaces:**
- Produces on `ToolRegistry`:
  - `build_cancellation(self, order_id: str, customer_id: str) -> CancellationPreview | str` — returns a preview when cancellable, or a refusal string (not found / already shipped / wrong state).
  - `commit_cancellation(self, preview: CancellationPreview) -> str` — runs the atomic cancel, returns the confirmation message; on `CancellationConflict` returns a safe message.
- Consumes: `OrderRepository.lookup`, `OrderRepository.cancel`, `CancellationConflict`, a `token_factory` (default `uuid4`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/workflow/test_tools.py (append)
from pwc_support.domain.models import CancellationPreview


def test_build_cancellation_previews_eligible_order(retail_db: Database) -> None:
    reg = _registry(retail_db)
    preview = reg.build_cancellation("ORD-2001", "CUS-1001")
    assert isinstance(preview, CancellationPreview)
    assert preview.order_id == "ORD-2001"
    # No mutation yet.
    order = OrderRepository(retail_db).lookup("ORD-2001", "CUS-1001")
    assert order is not None and order.status == "processing"


def test_build_cancellation_refuses_shipped_order(retail_db: Database) -> None:
    msg = _registry(retail_db).build_cancellation("ORD-5001", "CUS-1001")
    assert isinstance(msg, str) and "already shipped" in msg


def test_build_cancellation_refuses_unknown_order(retail_db: Database) -> None:
    msg = _registry(retail_db).build_cancellation("ORD-3001", "CUS-1001")
    assert isinstance(msg, str) and "could not find" in msg.lower()


def test_commit_cancellation_mutates_once(retail_db: Database) -> None:
    reg = _registry(retail_db)
    preview = reg.build_cancellation("ORD-2001", "CUS-1001")
    assert isinstance(preview, CancellationPreview)
    msg = reg.commit_cancellation(preview)
    assert msg == "Order ORD-2001 has been cancelled."
    order = OrderRepository(retail_db).lookup("ORD-2001", "CUS-1001")
    assert order is not None and order.status == "cancelled" and order.version == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/workflow/test_tools.py -q`
Expected: FAIL — `AttributeError: 'ToolRegistry' object has no attribute 'build_cancellation'`.

- [ ] **Step 3: Implement the helpers**

```python
# src/pwc_support/workflow/tools.py — extend imports and ToolRegistry
from collections.abc import Callable
from uuid import uuid4

from pwc_support.domain.models import CancellationPreview
from pwc_support.storage.retail_repositories import CancellationConflict

# Replace the existing ToolRegistry.__init__ with:
def __init__(self, products: ProductRepository, orders: OrderRepository, rag: RagAnswerer,
             token_factory: Callable[[], str] | None = None) -> None:
    self.products = products
    self.orders = orders
    self.rag = rag
    self.token_factory = token_factory or (lambda: str(uuid4()))

def build_cancellation(self, order_id: str, customer_id: str) -> CancellationPreview | str:
    order = self.orders.lookup(order_id, customer_id)
    if order is None:
        return "I could not find that order for this customer."
    if order.fulfilment_status != "processing" or order.status not in {"processing", "paid"}:
        reason = ("because it has already shipped"
                  if order.fulfilment_status in {"shipped", "delivered"} else "in its current state")
        return f"Order {order.order_id} cannot be cancelled {reason}."
    return CancellationPreview(
        confirmation_token=self.token_factory(), order_id=order.order_id, customer_id=customer_id,
        expected_version=order.version,
        summary=f"Cancel order {order.order_id} for {order.total} {order.currency}")

def commit_cancellation(self, preview: CancellationPreview) -> str:
    try:
        result = self.orders.cancel(preview)
    except CancellationConflict:
        return "The order could not be changed safely. No cancellation was made."
    return f"Order {result.order_id} has been cancelled."
```

- [ ] **Step 4: Run to verify pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/workflow/test_tools.py -q`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pwc_support/workflow/tools.py tests/unit/workflow/test_tools.py
git commit -m "feat: add cancellation preview and commit helpers to tool registry"
```

---

## Task 4: Agent graph — nodes, routing loop, confirm interrupt, memory

**Files:**
- Create: `src/pwc_support/workflow/agent_graph.py`
- Test: `tests/unit/workflow/test_agent_graph.py`

**Interfaces:**
- Consumes: `ToolRegistry`, `TOOL_SCHEMAS`, a model with `chat_with_tools(messages, tools) -> {"role","content","tool_calls"}`.
- Produces:
  - `class AgentState(TypedDict, total=False)`: `messages: Annotated[list[dict], add]`, `customer_id: str`, `steps: Annotated[list[str], add]`, `citations: tuple[Citation, ...]`, `iterations: int`, `response: str`.
  - `SYSTEM_PROMPT: str`.
  - `build_agent_graph(*, model, registry: ToolRegistry, max_iterations: int = 6) -> CompiledStateGraph` compiled with `MemorySaver()`.
- Behaviour: `intake` seeds a system message once + appends nothing else; `agent` calls the model; conditional edge routes to `confirm` when a `cancel_order` call is present, to `tools` for other tool calls, else to `respond`; `confirm` runs `interrupt({...})` and commits/rejects; `respond` sets `response` from the last assistant content and passes citations through.

- [ ] **Step 1: Write the failing tests** (use a scripted fake model; no Ollama)

```python
# tests/unit/workflow/test_agent_graph.py
from langgraph.types import Command

from pwc_support.rag.answer import RagAnswerer
from pwc_support.storage.database import Database
from pwc_support.storage.retail_repositories import OrderRepository, ProductRepository
from pwc_support.workflow.agent_graph import build_agent_graph
from pwc_support.workflow.tools import ToolRegistry
from tests.fakes import FakeGenerator, FakeKnowledgeBase


class ScriptedModel:
    """Return queued assistant messages, one per agent-node call."""
    def __init__(self, script: list[dict]) -> None:
        self.script = list(script)
        self.calls: list[list[dict]] = []

    def chat_with_tools(self, *, messages, tools):
        self.calls.append(list(messages))
        return self.script.pop(0)


def _registry(retail_db: Database) -> ToolRegistry:
    rag = RagAnswerer(FakeKnowledgeBase(), FakeGenerator())
    return ToolRegistry(ProductRepository(retail_db), OrderRepository(retail_db), rag)


def _config(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}}


def test_agent_calls_tool_then_answers(retail_db: Database) -> None:
    model = ScriptedModel([
        {"role": "assistant", "content": "",
         "tool_calls": [{"name": "get_order_status", "arguments": {"order_id": "ORD-5001"}}]},
        {"role": "assistant", "content": "Your order ORD-5001 is shipped.", "tool_calls": []},
    ])
    graph = build_agent_graph(model=model, registry=_registry(retail_db))
    out = graph.invoke({"messages": [{"role": "user", "content": "where is ORD-5001?"}],
                        "customer_id": "CUS-1001"}, _config("t1"))
    assert out["response"] == "Your order ORD-5001 is shipped."
    assert "get_order_status" in out["steps"]


def test_plain_answer_skips_tools(retail_db: Database) -> None:
    model = ScriptedModel([{"role": "assistant", "content": "Hello!", "tool_calls": []}])
    graph = build_agent_graph(model=model, registry=_registry(retail_db))
    out = graph.invoke({"messages": [{"role": "user", "content": "hi"}], "customer_id": "CUS-1001"},
                       _config("t2"))
    assert out["response"] == "Hello!"


def test_policy_answer_carries_citations(retail_db: Database) -> None:
    model = ScriptedModel([
        {"role": "assistant", "content": "",
         "tool_calls": [{"name": "search_policies", "arguments": {"question": "shipping time"}}]},
        {"role": "assistant", "content": "Standard shipping is three to five business days. [S1]",
         "tool_calls": []},
    ])
    graph = build_agent_graph(model=model, registry=_registry(retail_db))
    out = graph.invoke({"messages": [{"role": "user", "content": "shipping?"}],
                        "customer_id": "CUS-1001"}, _config("t3"))
    assert len(out["citations"]) >= 1


def test_cancellation_pauses_then_commits_on_yes(retail_db: Database) -> None:
    model = ScriptedModel([
        {"role": "assistant", "content": "",
         "tool_calls": [{"name": "cancel_order", "arguments": {"order_id": "ORD-2001"}}]},
        {"role": "assistant", "content": "Order ORD-2001 has been cancelled.", "tool_calls": []},
    ])
    graph = build_agent_graph(model=model, registry=_registry(retail_db))
    cfg = _config("t4")
    paused = graph.invoke({"messages": [{"role": "user", "content": "cancel ORD-2001"}],
                           "customer_id": "CUS-1001"}, cfg)
    assert paused["__interrupt__"][0].value["order_id"] == "ORD-2001"
    order = OrderRepository(retail_db).lookup("ORD-2001", "CUS-1001")
    assert order is not None and order.status == "processing"   # not yet mutated
    done = graph.invoke(Command(resume="yes"), cfg)
    assert "cancelled" in done["response"].lower()
    order = OrderRepository(retail_db).lookup("ORD-2001", "CUS-1001")
    assert order is not None and order.status == "cancelled" and order.version == 2


def test_cancellation_rejected_on_no(retail_db: Database) -> None:
    model = ScriptedModel([
        {"role": "assistant", "content": "",
         "tool_calls": [{"name": "cancel_order", "arguments": {"order_id": "ORD-4001"}}]},
        {"role": "assistant", "content": "Order ORD-4001 was not cancelled.", "tool_calls": []},
    ])
    graph = build_agent_graph(model=model, registry=_registry(retail_db))
    cfg = _config("t5")
    graph.invoke({"messages": [{"role": "user", "content": "cancel ORD-4001"}],
                  "customer_id": "CUS-1001"}, cfg)
    graph.invoke(Command(resume="no"), cfg)
    order = OrderRepository(retail_db).lookup("ORD-4001", "CUS-1001")
    assert order is not None and order.status == "paid" and order.version == 1


def test_max_iterations_guard_stops_loop(retail_db: Database) -> None:
    # Model always asks for a tool; guard must force a final answer.
    always = {"role": "assistant", "content": "",
              "tool_calls": [{"name": "search_products", "arguments": {"query": "x"}}]}
    model = ScriptedModel([always] * 20)
    graph = build_agent_graph(model=model, registry=_registry(retail_db), max_iterations=3)
    out = graph.invoke({"messages": [{"role": "user", "content": "loop"}], "customer_id": "CUS-1001"},
                       _config("t6"))
    assert out["response"]  # some safe non-empty response
    assert len(model.calls) <= 4
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/workflow/test_agent_graph.py -q`
Expected: FAIL — `ModuleNotFoundError: pwc_support.workflow.agent_graph`.

- [ ] **Step 3: Implement `agent_graph.py`**

```python
# src/pwc_support/workflow/agent_graph.py
from __future__ import annotations

from operator import add
from typing import Annotated, Any, Protocol, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from pwc_support.domain.models import Citation
from pwc_support.workflow.tools import TOOL_SCHEMAS, ToolRegistry

SYSTEM_PROMPT = (
    "You are a retail customer support agent. Use the tools to answer questions about "
    "products, offers, orders, and policies. Only answer policy questions from the "
    "search_policies tool result, keeping its [S1] citation markers verbatim; never state "
    "policy from your own knowledge. To cancel an order, call cancel_order; the system will "
    "ask the customer to confirm. Be concise."
)


class ToolModel(Protocol):
    def chat_with_tools(self, *, messages: list[dict], tools: list[dict]) -> dict[str, Any]: ...


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict[str, Any]], add]
    customer_id: str
    steps: Annotated[list[str], add]
    citations: tuple[Citation, ...]
    iterations: int
    response: str


def build_agent_graph(
    *, model: ToolModel, registry: ToolRegistry, max_iterations: int = 6
) -> CompiledStateGraph[Any, Any, Any, Any]:
    def intake(state: AgentState) -> dict[str, Any]:
        has_system = any(m.get("role") == "system" for m in state.get("messages", []))
        if has_system:
            return {}
        return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}]}

    def agent(state: AgentState) -> dict[str, Any]:
        message = model.chat_with_tools(messages=list(state["messages"]), tools=TOOL_SCHEMAS)
        return {"messages": [message], "iterations": state.get("iterations", 0) + 1}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        calls = last.get("tool_calls") or []
        if not calls or state.get("iterations", 0) >= max_iterations:
            return "respond"
        if any(c["name"] == "cancel_order" for c in calls):
            return "confirm"
        return "tools"

    def tools(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        results: list[dict[str, Any]] = []
        steps: list[str] = []
        citations: tuple[Citation, ...] = state.get("citations", ())
        for call in last.get("tool_calls") or []:
            outcome = registry.run(call["name"], call["arguments"], customer_id=state["customer_id"])
            results.append({"role": "tool", "tool_name": call["name"], "content": outcome.content})
            steps.append(call["name"])
            if outcome.citations:
                citations = outcome.citations
        return {"messages": results, "steps": steps, "citations": citations}

    def confirm(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        call = next(c for c in last["tool_calls"] if c["name"] == "cancel_order")
        preview = registry.build_cancellation(str(call["arguments"].get("order_id", "")),
                                              state["customer_id"])
        if isinstance(preview, str):
            content = preview
        else:
            decision = interrupt({"order_id": preview.order_id, "summary": preview.summary})
            content = (registry.commit_cancellation(preview)
                       if str(decision).strip().lower() in {"yes", "y", "confirm"}
                       else f"Order {preview.order_id} was not cancelled.")
        return {"messages": [{"role": "tool", "tool_name": "cancel_order", "content": content}],
                "steps": ["cancel_order"]}

    def respond(state: AgentState) -> dict[str, Any]:
        answers = [m.get("content", "") for m in state["messages"] if m.get("role") == "assistant"]
        text = answers[-1] if answers and answers[-1] else "I'm not sure how to help with that."
        return {"response": text, "citations": state.get("citations", ())}

    builder: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
    for name, fn in [("intake", intake), ("agent", agent), ("tools", tools),
                     ("confirm", confirm), ("respond", respond)]:
        builder.add_node(name, fn)
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "agent")
    builder.add_conditional_edges("agent", route)
    builder.add_edge("tools", "agent")
    builder.add_edge("confirm", "agent")
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=MemorySaver())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/workflow/test_agent_graph.py -q`
Expected: PASS (6 tests). If the `confirm` re-run on resume double-builds a preview, that is expected and harmless (version check guards the commit).

- [ ] **Step 5: Commit**

```bash
git add src/pwc_support/workflow/agent_graph.py tests/unit/workflow/test_agent_graph.py
git commit -m "feat: build custom tool-calling agent graph with confirm interrupt"
```

---

## Task 5: `AgentService` (submit/resume) + `ChatReply` fields

**Files:**
- Modify: `src/pwc_support/domain/models.py` (extend `ChatReply`)
- Rewrite: `src/pwc_support/services/chat.py`
- Test: `tests/unit/services/test_chat.py`

**Interfaces:**
- Produces:
  - `ChatReply` gains: `steps: tuple[str, ...] = ()`, `awaiting_confirmation: bool = False`, `preview: str = ""`. Keep `message`, `citations`, `total_duration_ms`; drop `pending_cancellation`, `awaiting_cancel`, `tasks`, `events` if unused elsewhere (keep `tasks`/`events` optional to avoid churn — set to `()`).
  - `class AgentService.__init__(self, graph)`; `submit(self, *, thread_id, body, customer_id) -> ChatReply`; `resume(self, *, thread_id, decision) -> ChatReply`.
  - A helper `_project(terminal_or_interrupt) -> ChatReply` handling both the interrupted state (`__interrupt__` present → `awaiting_confirmation=True`, `preview=<summary>`) and the terminal state.
- Consumes: `graph.invoke(input, config)` and `graph.invoke(Command(resume=...), config)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/services/test_chat.py  (replace file contents)
from typing import Any

from langgraph.types import Command
from pwc_support.services.chat import AgentService


class FakeGraph:
    def __init__(self) -> None:
        self.invocations: list[Any] = []

    def invoke(self, payload, config=None):
        self.invocations.append((payload, config))
        if isinstance(payload, Command):
            return {"response": "Order ORD-2001 has been cancelled.", "citations": (), "steps": ["cancel_order"]}
        text = payload["messages"][-1]["content"]
        if "cancel" in text:
            class _I:  # mimic Interrupt object
                value = {"order_id": "ORD-2001", "summary": "Cancel order ORD-2001 for 79.99 EUR"}
            return {"__interrupt__": [_I()]}
        return {"response": "Order ORD-5001 is shipped.", "citations": (), "steps": ["get_order_status"]}


def test_submit_projects_answer() -> None:
    reply = AgentService(FakeGraph()).submit(thread_id="t", body="where is ORD-5001?", customer_id="CUS-1001")
    assert reply.message == "Order ORD-5001 is shipped."
    assert reply.awaiting_confirmation is False
    assert reply.steps == ("get_order_status",)


def test_submit_surfaces_confirmation_interrupt() -> None:
    reply = AgentService(FakeGraph()).submit(thread_id="t", body="cancel ORD-2001", customer_id="CUS-1001")
    assert reply.awaiting_confirmation is True
    assert "Cancel order ORD-2001" in reply.preview


def test_resume_projects_final_answer() -> None:
    graph = FakeGraph()
    reply = AgentService(graph).resume(thread_id="t", decision="yes")
    assert "cancelled" in reply.message.lower()
    assert isinstance(graph.invocations[-1][0], Command)


def test_submit_handles_graph_failure() -> None:
    class Boom:
        def invoke(self, *_a, **_k):
            raise RuntimeError("offline")
    reply = AgentService(Boom()).submit(thread_id="t", body="hi", customer_id="CUS-1001")
    assert reply.message == "The support agent is unavailable. Please try again."
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/services/test_chat.py -q`
Expected: FAIL — `ImportError: cannot import name 'AgentService'`.

- [ ] **Step 3: Extend `ChatReply` and implement `AgentService`**

```python
# src/pwc_support/domain/models.py — ChatReply
# Keep tasks/events/pending_cancellation as deprecated optional fields so app.py,
# the eval script, and existing tests don't break before Task 6–7 update them.
class ChatReply(DomainModel):
    message: str
    citations: tuple[Citation, ...] = ()
    steps: tuple[str, ...] = ()
    awaiting_confirmation: bool = False
    preview: str = ""
    tasks: tuple[Task, ...] = ()                                   # deprecated — remove in Task 7
    events: tuple[TraceEvent, ...] = ()                            # deprecated — remove in Task 7
    pending_cancellation: CancellationPreview | None = None        # deprecated — remove in Task 7
    awaiting_cancel: bool = False                                  # deprecated — remove in Task 7
    total_duration_ms: float = Field(ge=0, default=0.0)
```

```python
# src/pwc_support/services/chat.py  (full rewrite)
from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from langgraph.types import Command

from pwc_support.domain.models import ChatReply

logger = logging.getLogger(__name__)


class AgentGraph(Protocol):
    def invoke(self, payload: Any, config: dict[str, Any] | None = None) -> dict[str, Any]: ...


class AgentService:
    def __init__(self, graph: AgentGraph) -> None:
        self.graph = graph

    def submit(self, *, thread_id: str, body: str, customer_id: str) -> ChatReply:
        payload = {"messages": [{"role": "user", "content": body}], "customer_id": customer_id}
        return self._run(payload, thread_id)

    def resume(self, *, thread_id: str, decision: str) -> ChatReply:
        return self._run(Command(resume=decision), thread_id)

    def _run(self, payload: Any, thread_id: str) -> ChatReply:
        started = time.perf_counter()
        config = {"configurable": {"thread_id": thread_id}}
        try:
            terminal = self.graph.invoke(payload, config)
        except Exception:
            logger.exception("agent failed", extra={"thread_id": thread_id})
            return ChatReply(message="The support agent is unavailable. Please try again.",
                             total_duration_ms=self._ms(started))
        interrupts = terminal.get("__interrupt__")
        if interrupts:
            value = interrupts[0].value
            return ChatReply(message=value.get("summary", "") + "? Please answer yes or no.",
                             awaiting_confirmation=True, preview=value.get("summary", ""),
                             total_duration_ms=self._ms(started))
        return ChatReply(message=str(terminal.get("response", "")),
                         citations=tuple(terminal.get("citations", ())),
                         steps=tuple(terminal.get("steps", ())),
                         total_duration_ms=self._ms(started))

    @staticmethod
    def _ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)
```

- [ ] **Step 4: Run to verify pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/services/test_chat.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Update `services/__init__.py`**

```python
# src/pwc_support/services/__init__.py
"""Application services exposed to UI and channel adapters."""

from pwc_support.services.chat import AgentService

__all__ = ["AgentService"]
```

- [ ] **Step 6: Commit**

```bash
git add src/pwc_support/domain/models.py src/pwc_support/services/chat.py \
     src/pwc_support/services/__init__.py tests/unit/services/test_chat.py
git commit -m "feat: add AgentService with submit/resume over the agent graph"
```

---

## Task 6: Wire bootstrap + Streamlit UI

**Files:**
- Modify: `src/pwc_support/bootstrap.py`
- Modify: `app.py`
- Test: `tests/ui/test_retail_contracts.py` (adjust to new Runtime/service surface)

**Interfaces:**
- Consumes: `build_agent_graph`, `ToolRegistry`, `AgentService`.
- Produces: `Runtime` with `service: AgentService`, `settings`, `knowledge_base`. `build_runtime()` builds the registry + agent graph.

- [ ] **Step 1: Update `bootstrap.py`** — replace imports, `Runtime` type, and graph/service construction

```python
# src/pwc_support/bootstrap.py — full rewrite of imports and wiring
# Replace these imports:
#   from pwc_support.services.chat import ChatService
#   from pwc_support.workflow.commerce import CommerceTools
#   from pwc_support.workflow.graph import build_graph
#   from pwc_support.workflow.planner import OllamaPlanner
# With:
from pwc_support.services.chat import AgentService
from pwc_support.workflow.agent_graph import build_agent_graph
from pwc_support.workflow.tools import ToolRegistry

# Update the Runtime dataclass:
@dataclass(frozen=True, slots=True)
class Runtime:
    settings: Settings
    service: AgentService          # was ChatService
    database: Database
    knowledge_base: ChromaKnowledgeBase

# In build_runtime(), replace the graph/service construction block:
registry = ToolRegistry(ProductRepository(database), OrderRepository(database), rag_answerer)
graph = build_agent_graph(model=model, registry=registry)
return Runtime(settings=resolved, service=AgentService(graph), database=database,
               knowledge_base=knowledge_base)
```

- [ ] **Step 2: Update `app.py`** — thread id, resume routing, steps trace, render_trace rewrite

```python
# app.py — add uuid import at the top
import uuid

# app.py — session defaults (replace initialise_session body)
defaults = {"messages": [], "customer_id": "CUS-1001",
            "thread_id": str(uuid.uuid4()), "awaiting_confirmation": False}

# app.py — replace render_trace to use reply.steps instead of reply.events/tasks
def render_trace(reply: ChatReply) -> None:
    label = f"Trace: {len(reply.steps)} steps, {reply.total_duration_ms:.0f} ms"
    with st.expander(label):
        if reply.steps:
            st.caption("Tools used: " + ", ".join(reply.steps))
        if reply.awaiting_confirmation:
            st.info(f"⏸ Awaiting confirmation: {reply.preview}")

# app.py — replace the chat_input handler block
if question := st.chat_input("Ask about products, offers, policies, or your order"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Running locally"):
        if st.session_state.awaiting_confirmation:
            reply = runtime.service.resume(thread_id=st.session_state.thread_id, decision=question)
        else:
            reply = runtime.service.submit(thread_id=st.session_state.thread_id,
                                           body=question, customer_id=st.session_state.customer_id)
    st.session_state.awaiting_confirmation = reply.awaiting_confirmation
    st.session_state.messages.append(
        {"role": "assistant", "content": reply.message, "reply": reply}
    )
    st.rerun()
```

- [ ] **Step 3: Replace the UI contract test**

```python
# tests/ui/test_retail_contracts.py (full rewrite)
from pathlib import Path


def test_ui_is_one_natural_language_chat() -> None:
    source = Path("app.py").read_text(encoding="utf-8")

    assert source.count("st.chat_input") == 1
    assert "st.tabs" not in source
    assert "Simulated email" not in source
    assert "Human review" not in source


def test_ui_routes_confirmation_via_resume() -> None:
    source = Path("app.py").read_text(encoding="utf-8")

    assert '"awaiting_confirmation"' in source
    assert "reply.awaiting_confirmation" in source
    assert "runtime.service.resume" in source
    assert "thread_id" in source
```

- [ ] **Step 4: Manual smoke + run suite**

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider
```
Expected: PASS. Then a manual Streamlit smoke (seed + ingest + run) per the Global Constraints commands.

- [ ] **Step 5: Commit**

```bash
git add src/pwc_support/bootstrap.py app.py tests/ui/test_retail_contracts.py
git commit -m "feat: wire agent service into runtime and Streamlit UI"
```

---

## Task 7: Remove the planner and old graph

**Files:**
- Delete: `src/pwc_support/workflow/planner.py`, `src/pwc_support/workflow/graph.py`, `src/pwc_support/workflow/commerce.py` (if no longer imported)
- Delete: `tests/unit/workflow/test_planner.py`, `tests/unit/workflow/test_graph.py`, `tests/unit/workflow/test_commerce.py`
- Modify: `src/pwc_support/domain/state.py` (remove `SupportState` if unused), any lingering imports

- [ ] **Step 1: Grep for references**

```bash
rg -n "planner|SupportState|build_graph|CommerceTools|plan_tasks" src app.py scripts tests
```
Resolve every hit: repositories/RagAnswerer stay; planner/graph/commerce references go.

- [ ] **Step 2: Delete files and dead tests; fix imports**

- [ ] **Step 3: Remove deprecated `ChatReply` fields**

Now that `app.py`, the eval script, and all tests use the new fields (`steps`, `awaiting_confirmation`, `preview`), remove the deprecated fields from `ChatReply` in `src/pwc_support/domain/models.py`:

```python
# src/pwc_support/domain/models.py — final ChatReply (remove deprecated lines)
class ChatReply(DomainModel):
    message: str
    citations: tuple[Citation, ...] = ()
    steps: tuple[str, ...] = ()
    awaiting_confirmation: bool = False
    preview: str = ""
    total_duration_ms: float = Field(ge=0, default=0.0)
```

Also remove the now-unused imports: `Task`, `TraceEvent`, `CancellationPreview` (if no other model uses them — `CancellationPreview` is still used by `retail_repositories.py`, so keep it; only drop `Task` and `TraceEvent` from `ChatReply`'s perspective).

- [ ] **Step 4: Run full suite + gates**

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/ruff format src scripts tests app.py && .venv/bin/ruff check src scripts tests app.py
.venv/bin/mypy src scripts app.py --cache-dir /private/tmp/customer-support-mypy
```
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: remove planner and deterministic router"
```

---

## Task 8: Re-point the evaluation to intent checks

**Files:**
- Modify: `scripts/run_evaluation.py`
- Modify: `tests/unit/scripts/test_evaluation.py`
- Keep: `eval/final.jsonl` (the 16 cases), extend each case with a `expected_tool` and keep `required_sources`, `required_terms` (as substring hints), `expect_no_hallucination`.

**Interfaces:**
- The runner now drives `AgentService.submit(...)` with a fresh `thread_id` per case (and a follow-up `resume("yes")` for cancel-confirm cases), then scores: `routing` = expected tool appears in `reply.steps`; `sources`/`attribution` = citations present when required; `safety` = cancellation gated (first turn `awaiting_confirmation` for cancel cases) and no cross-customer leak; `terms` = required substrings present.

- [ ] **Step 1: Update the eval runner and scorer**

Key changes to `scripts/run_evaluation.py`:

```python
# scripts/run_evaluation.py — key changes
# 1. Import AgentService instead of ChatService; import uuid
import uuid
from pwc_support.services.chat import AgentService

# 2. Replace the per-case loop body:
for index, case in enumerate(cases, start=1):
    customer_id = str(case.get("customer_id", "CUS-1001"))
    thread_id = str(uuid.uuid4())
    reply: ChatReply | None = None
    turns = case.get("turns", [case.get("question", "")])
    for turn in turns:
        if reply is not None and reply.awaiting_confirmation:
            # This turn is the yes/no confirmation
            reply = service.resume(thread_id=thread_id, decision=str(turn))
        else:
            reply = service.submit(thread_id=thread_id, body=str(turn),
                                   customer_id=customer_id)
    assert reply is not None
    score = score_case(case, reply)
    # ...

# 3. Update score_case to check reply.steps instead of reply.tasks:
def score_case(case: dict[str, Any], reply: ChatReply) -> CaseScore:
    text = reply.message.casefold()
    expected_tools = set(case.get("expected_tools", []))
    actual_steps = set(reply.steps)
    # ...
    checks = {
        "routing": expected_tools <= actual_steps,
        "sources": required_sources <= actual_sources,
        "terms": all(term.casefold() in text for term in case.get("required_terms", [])),
        "safety": all(term.casefold() not in text for term in case.get("forbidden_terms", []))
                  and (not case.get("expect_confirmation", False)
                       or reply.awaiting_confirmation == case["expect_confirmation"]),
        "attribution": ...,  # unchanged
    }
```

Update `eval/final.jsonl` cases: replace `"expected_task_kinds"` with `"expected_tools"` (e.g. `["get_order_status"]`, `["search_policies"]`, `["cancel_order"]`), replace `"expect_pending_cancellation"` with `"expect_confirmation"`, and add multi-turn cancellation cases as `"turns": ["cancel ORD-2001", "yes"]`.

- [ ] **Step 2: Write a unit test for the new scorer**

```python
# tests/unit/scripts/test_evaluation.py — verify new scoring on a 2-case subset
from pwc_support.domain.models import ChatReply
from scripts.run_evaluation import score_case

def test_tool_routing_scores_correctly() -> None:
    case = {"id": "order-status", "expected_tools": ["get_order_status"],
            "required_terms": ["shipped"], "required_sources": [], "forbidden_terms": []}
    reply = ChatReply(message="Order ORD-5001 is shipped.",
                      steps=("get_order_status",), total_duration_ms=10.0)
    score = score_case(case, reply)
    assert score.checks["routing"] is True
    assert score.checks["terms"] is True

def test_cancel_safety_requires_confirmation_gate() -> None:
    case = {"id": "cancel-safe", "expected_tools": ["cancel_order"],
            "expect_confirmation": True, "required_terms": [], "required_sources": [],
            "forbidden_terms": []}
    reply = ChatReply(message="Cancel order ORD-2001 for 79.99 EUR? Please answer yes or no.",
                      awaiting_confirmation=True, steps=("cancel_order",),
                      total_duration_ms=10.0)
    score = score_case(case, reply)
    assert score.checks["safety"] is True
```

- [ ] **Step 3: Run `pytest tests/unit/scripts/test_evaluation.py`; then the live eval** per Global Constraints and confirm ≥ 90% with safety 100%.
- [ ] **Step 4: Commit**

```bash
git add scripts/run_evaluation.py tests/unit/scripts/test_evaluation.py eval/final.jsonl
git commit -m "test: evaluate agent by tool-intent and grounding"
```

---

## Task 9: Update README and spec cross-links

**Files:**
- Modify: `README.md` (architecture section, node list, agent/tool description, eval + load results, run instructions)

- [ ] **Step 1: Rewrite the Architecture section** to describe the 5-node agent graph + RAG subgraph + the five tools + confirm interrupt + in-memory checkpointer.
- [ ] **Step 2: Refresh evaluation and load numbers** from the re-run artifacts.
- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document the tool-calling agent architecture"
```

---

## Self-Review

**Spec coverage:** every spec section maps to a task — gateway tool-calling (T1), tools + grounding + scoping (T2), cancellation preview/commit (T3), 5-node graph + routing + interrupt + memory (T4), service submit/resume + reply fields (T5), bootstrap + UI + steps trace (T6), deletion of planner/router (T7), intent eval (T8), README (T9). Requirements-mapping table in the spec is covered by T4 (nodes/routing/state/subgraph), T2 (tools), T6 (UI), T8 (eval), and the untouched Dockerfile.

**Placeholder scan:** code steps carry real code using verified signatures (`ProductRepository.search/list_active_offers`, `OrderRepository.lookup/cancel`, `RagAnswerer.answer`, `CancellationPreview` fields, Ollama tool-call shape). T6/T8/T9 describe edits against files whose exact current contents the implementer must open; their code fragments are concrete but bounded to the changed lines.

**Type consistency:** `ToolOutcome`, `ToolRegistry.run`, `build_cancellation`/`commit_cancellation`, `AgentState`, `build_agent_graph`, `AgentService.submit/resume`, and `ChatReply` fields are named identically across the tasks that define and consume them.

**Post-review fixes applied:**
- T3: `token_factory` shown as full `__init__` code (was only a comment).
- T5: `ChatReply` keeps deprecated `tasks`/`events`/`pending_cancellation` as optional fields until T7 removes them — prevents intermediate breakage.
- T5: `services/__init__.py` update added (was missing — would break `from pwc_support.services import ChatService`).
- T6: Full `Runtime` dataclass update shown (type annotation `service: AgentService`, old imports replaced).
- T6: Concrete `test_retail_contracts.py` replacement provided (was described but had no code).
- T7: Explicit step to remove deprecated `ChatReply` fields added.
- T8: Concrete eval script rewrite with `resume()` flow for cancel cases.

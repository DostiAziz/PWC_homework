# Agentic RAG Customer Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local PwC-style client-support prototype that automatically answers grounded general questions, pauses defined cases for human review, resumes the same conversation, and satisfies every evaluation and packaging requirement in the supplied proposal.

**Architecture:** A Streamlit application calls a typed application service backed by an eight-node LangGraph workflow. The workflow invokes a separate four-node Chroma/Nomic RAG subgraph, typed SQLite case and simulated-mailbox tools, and persistent LangGraph interrupts; GPT-OSS and Nomic run through native macOS Ollama.

**Tech Stack:** Python 3.12, uv, Pydantic v2, LangGraph 1.2.11, SQLite checkpointer 3.1.1, Ollama Python 0.6.2, `gpt-oss:20b`, `nomic-embed-text`, Chroma 1.5.9, Streamlit 1.63.0, pytest 9.1.1, Ruff 0.16.5, mypy 2.3.1, Docker and Docker Compose.

**Spec:** `docs/superpowers/specs/2026-09-02-agentic-rag-customer-support-design.md`

## Global Constraints

- Use Python `>=3.12,<3.13`; keep all direct dependencies at the exact versions in this plan and commit `uv.lock`.
- Run generation with native Ollama model `gpt-oss:20b`; run embeddings with `nomic-embed-text`; never download or change a model implicitly at application startup.
- Default generation context is 8,192 tokens; answer output is at most 512 tokens; triage/planning output is at most 256 tokens; request timeout is 120 seconds.
- Permit one concurrent generation, one transport retry, at most four planned tasks, and at most two composition revisions.
- Use English (`en`) in version 1 while retaining language metadata.
- Use financial services as the main domain, with limited healthcare and technology examples.
- Automatically release supported general answers after deterministic verification.
- Require human review for confidentiality/cybersecurity incidents, legal/regulatory advice, complaints/escalations, external actions/commitments, insufficient/conflicting evidence, and engagement-specific professional judgement.
- Use only public attributed summaries and clearly labelled synthetic cases; do not imply access to PwC internal data or actual clients.
- Persist operational data and checkpoints in separate SQLite databases; make all side effects transactional and idempotent.
- Display operational events, routing, evidence, and node timing; never expose chain-of-thought, secrets, full private messages in logs, or untrusted source instructions.
- Keep Ollama native on macOS for Apple GPU acceleration; containerize the application and Chroma HTTP deployment.
- Do not report functional or performance results until the real commands have run and raw artifacts exist.

---

## Planned File Structure

```text
.
├── .dockerignore
├── .env.example
├── .gitignore
├── .python-version
├── Dockerfile
├── README.md
├── compose.yaml
├── pyproject.toml
├── streamlit_app.py
├── uv.lock
├── config/
│   └── retrieval.json
├── corpus/
│   ├── manifest.json
│   └── documents/
│       ├── pwc-financial-services.md
│       ├── pwc-global-services.md
│       ├── pwc-industries.md
│       ├── pwc-network-structure.md
│       └── synthetic-support-faq.md
├── data/
│   ├── chroma/.gitkeep
│   └── state/.gitkeep
├── artifacts/
│   ├── evaluation/.gitkeep
│   ├── events/.gitkeep
│   └── load/.gitkeep
├── eval/
│   ├── development.jsonl
│   ├── final.jsonl
│   └── load_workload.jsonl
├── scripts/
│   ├── calibrate_retrieval.py
│   ├── check_runtime.py
│   ├── ingest_corpus.py
│   ├── run_evaluation.py
│   ├── run_load.py
│   └── seed_demo.py
├── src/pwc_support/
│   ├── __init__.py
│   ├── bootstrap.py
│   ├── config.py
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── errors.py
│   │   ├── models.py
│   │   └── state.py
│   ├── ports/
│   │   ├── __init__.py
│   │   ├── cases.py
│   │   ├── knowledge.py
│   │   ├── llm.py
│   │   └── mailbox.py
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── chroma.py
│   │   ├── ollama.py
│   │   ├── simulated_mailbox.py
│   │   └── sqlite/
│   │       ├── __init__.py
│   │       ├── database.py
│   │       ├── repositories.py
│   │       └── schema.sql
│   ├── observability/
│   │   ├── __init__.py
│   │   └── events.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── chunking.py
│   │   ├── graph.py
│   │   └── ingestion.py
│   ├── workflow/
│   │   ├── __init__.py
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   ├── policy.py
│   │   ├── reducers.py
│   │   └── routing.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── client_support.py
│   │   └── review.py
│   └── ui/
│       ├── __init__.py
│       ├── app.py
│       ├── client.py
│       ├── email.py
│       ├── review.py
│       └── trace.py
└── tests/
    ├── conftest.py
    ├── contract/
    ├── integration/
    └── unit/
```

Generated databases, Chroma vectors, event logs, and measured result files stay out of Git. Empty artifact directories use `.gitkeep`; final result summaries and selected raw JSON outputs are added explicitly after evaluation.

### Task 1: Project Foundation and Validated Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/pwc_support/__init__.py`
- Create: `src/pwc_support/config.py`
- Create: `tests/unit/test_config.py`
- Create: `uv.lock`

**Interfaces:**
- Consumes: environment variables documented in `.env.example`.
- Produces: `Settings.from_env() -> Settings` and exact locked dependencies used by every later task.

- [ ] **Step 1: Write the failing configuration tests**

```python
# tests/unit/test_config.py
from pathlib import Path

import pytest
from pydantic import ValidationError

from pwc_support.config import Settings


def test_defaults_match_mac_profile(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    assert settings.generation_model == "gpt-oss:20b"
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.num_ctx == 8192
    assert settings.max_parallel_generations == 1
    assert settings.operations_db == tmp_path / "state" / "operations.sqlite3"


def test_version_one_rejects_non_english() -> None:
    with pytest.raises(ValidationError):
        Settings(supported_language="de")
```

- [ ] **Step 2: Run the test and verify the import failure**

Run: `uv run pytest tests/unit/test_config.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'pwc_support'`.

- [ ] **Step 3: Add exact project metadata and dependencies**

```toml
# pyproject.toml
[project]
name = "pwc-agentic-support"
version = "0.1.0"
description = "Local Agentic RAG customer-support prototype"
requires-python = ">=3.12,<3.13"
dependencies = [
  "chromadb==1.5.9",
  "langgraph==1.2.11",
  "langgraph-checkpoint-sqlite==3.1.1",
  "ollama==0.6.2",
  "pydantic==2.13.5",
  "streamlit==1.63.0",
]

[dependency-groups]
dev = [
  "mypy==2.3.1",
  "pytest==9.1.1",
  "ruff==0.16.5",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/pwc_support"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["integration: requires local Ollama or Chroma"]

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]

[tool.mypy]
python_version = "3.12"
strict = true
mypy_path = "src"
```

Write `3.12` to `.python-version`. Add `.venv/`, `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `data/chroma/*`, `data/state/*`, `artifacts/events/*.jsonl`, and unselected evaluation/load run files to `.gitignore`, preserving each directory's `.gitkeep`.

- [ ] **Step 4: Implement validated settings**

```python
# src/pwc_support/config.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Settings(BaseModel):
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    ollama_base_url: str = "http://127.0.0.1:11434"
    generation_model: str = "gpt-oss:20b"
    embedding_model: str = "nomic-embed-text"
    chroma_mode: Literal["persistent", "http"] = "persistent"
    chroma_host: str = "127.0.0.1"
    chroma_port: int = Field(default=8000, ge=1, le=65535)
    collection_name: str = "pwc_support_v1_nomic_768"
    supported_language: Literal["en"] = "en"
    num_ctx: int = Field(default=8192, ge=2048, le=32768)
    answer_tokens: int = Field(default=512, ge=64, le=1024)
    schema_tokens: int = Field(default=256, ge=64, le=512)
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    max_parallel_generations: int = Field(default=1, ge=1, le=2)
    embedding_batch_size: int = Field(default=16, ge=1, le=64)
    max_planned_tasks: int = Field(default=4, ge=1, le=4)
    max_revisions: int = Field(default=2, ge=0, le=2)

    @property
    def operations_db(self) -> Path:
        return self.data_dir / "state" / "operations.sqlite3"

    @property
    def checkpoints_db(self) -> Path:
        return self.data_dir / "state" / "checkpoints.sqlite3"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"

    @model_validator(mode="after")
    def protect_mac_profile(self) -> "Settings":
        if self.max_parallel_generations > 1 and self.num_ctx > 8192:
            raise ValueError("parallel generation above one requires num_ctx <= 8192")
        return self

    @classmethod
    def from_env(cls) -> "Settings":
        values = {
            "data_dir": os.getenv("PWC_DATA_DIR", "data"),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            "chroma_mode": os.getenv("CHROMA_MODE", "persistent"),
            "chroma_host": os.getenv("CHROMA_HOST", "127.0.0.1"),
            "chroma_port": int(os.getenv("CHROMA_PORT", "8000")),
        }
        return cls.model_validate(values)
```

Create `.env.example` with the five environment variables read above and the container value `OLLAMA_BASE_URL=http://host.docker.internal:11434` as a comment.

- [ ] **Step 5: Lock dependencies and run the foundation gates**

Run:

```bash
uv lock
uv sync --all-groups
uv run pytest tests/unit/test_config.py -q
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src tests
```

Expected: two tests pass and all static checks exit `0`.

- [ ] **Step 6: Commit the foundation**

```bash
git add pyproject.toml uv.lock .python-version .gitignore .env.example src tests
git commit -m "build: initialize typed Python project"
```

### Task 2: Domain Models, Graph State, and Port Contracts

**Files:**
- Create: `src/pwc_support/domain/errors.py`
- Create: `src/pwc_support/domain/models.py`
- Create: `src/pwc_support/domain/state.py`
- Create: `src/pwc_support/ports/llm.py`
- Create: `src/pwc_support/ports/knowledge.py`
- Create: `src/pwc_support/ports/cases.py`
- Create: `src/pwc_support/ports/mailbox.py`
- Create: `tests/unit/domain/test_models.py`
- Create: `tests/unit/workflow/test_reducers.py`

**Interfaces:**
- Consumes: `Settings` from Task 1.
- Produces: all Pydantic boundary types; `SupportState`; `merge_task_results`; `LLMPort`, `KnowledgePort`, `CasePort`, and `MailboxPort` protocols.

- [ ] **Step 1: Write failing model and reducer tests**

```python
# tests/unit/domain/test_models.py
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from pwc_support.domain.models import Channel, IncomingMessage, TaskResult, TaskStatus


def test_incoming_message_normalizes_body() -> None:
    message = IncomingMessage(
        message_id=uuid4(), conversation_id=uuid4(), channel=Channel.CHAT,
        provider="local_chat", provider_message_id="m-1", provider_thread_id="t-1",
        sender_id="client-1", recipient_id="support", body="  Hello   there  ",
        language="en", received_at=datetime.now(UTC),
    )
    assert message.body == "Hello there"


def test_message_rejects_oversized_body() -> None:
    with pytest.raises(ValidationError):
        IncomingMessage.model_validate({"body": "x" * 8001})
```

```python
# tests/unit/workflow/test_reducers.py
import pytest

from pwc_support.domain.models import TaskResult, TaskStatus
from pwc_support.workflow.reducers import merge_task_results


def test_reducer_rejects_incompatible_duplicate() -> None:
    first = TaskResult(task_id="task-1", status=TaskStatus.SUCCESS, payload={"a": 1})
    second = TaskResult(task_id="task-1", status=TaskStatus.SUCCESS, payload={"a": 2})
    with pytest.raises(ValueError, match="conflicting task result"):
        merge_task_results({"task-1": first}, {"task-1": second})
```

- [ ] **Step 2: Run tests and verify missing domain modules**

Run: `uv run pytest tests/unit/domain/test_models.py tests/unit/workflow/test_reducers.py -q`

Expected: collection fails because `pwc_support.domain.models` does not exist.

- [ ] **Step 3: Define enums and boundary models**

Implement `domain/models.py` with string enums `Channel`, `OutcomeStatus`, `Route`, `ReviewCategory`, `TaskKind`, `TaskStatus`, `ReviewDecisionKind`, and `CaseStatus`. Define the models below with `extra="forbid"`, UTC datetimes, bounded strings, and immutable IDs:

```python
class IncomingMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message_id: UUID
    conversation_id: UUID
    channel: Channel
    provider: Literal["local_chat", "simulated_email"]
    provider_message_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    provider_thread_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    sender_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    recipient_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    subject: Annotated[str, StringConstraints(max_length=300)] | None = None
    body: Annotated[str, StringConstraints(min_length=1, max_length=8000)]
    language: Literal["en"]
    received_at: datetime
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("body")
    @classmethod
    def normalize_body(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("body cannot be blank")
        return normalized


class Citation(BaseModel):
    source_id: str
    chunk_id: str
    marker: str
    title: str
    canonical_url: HttpUrl | None
    heading: str
    excerpt: str
    similarity: float = Field(ge=-1.0, le=1.0)


class ClientOutcome(BaseModel):
    conversation_id: UUID
    case_id: str | None = None
    status: OutcomeStatus
    message: str
    citations: tuple[Citation, ...] = ()
    workflow_run_id: UUID
```

Also define the spec models `TriageDecision`, `PlannedTask`, `WorkPlan`, `TaskResult`, `RetrievalHit`, `RagRequest`, `RagResult`, `DraftReply`, `ProposedAction`, `CaseRecord`, `CaseLookupResult`, `ReviewRequest`, `ReviewDecision`, `ConversationView`, and `OperationalEvent`. Enforce one-to-four tasks in `WorkPlan`, require edited text for `EDIT`, reject edited text for `APPROVE`, and require matching client/case identifiers in case requests.

- [ ] **Step 4: Define stable errors and state**

```python
# src/pwc_support/domain/errors.py
from enum import StrEnum


class ErrorCode(StrEnum):
    INVALID_INPUT = "INVALID_INPUT"
    UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"
    DUPLICATE_MESSAGE = "DUPLICATE_MESSAGE"
    OLLAMA_UNAVAILABLE = "OLLAMA_UNAVAILABLE"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
    POLICY_REVIEW_REQUIRED = "POLICY_REVIEW_REQUIRED"
    CASE_ACCESS_DENIED = "CASE_ACCESS_DENIED"
    VERSION_CONFLICT = "VERSION_CONFLICT"
    CHECKPOINT_FAILED = "CHECKPOINT_FAILED"
    DELIVERY_FAILED = "DELIVERY_FAILED"


class SupportError(RuntimeError):
    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
```

Define `SupportState(TypedDict, total=False)` in `domain/state.py`. Use serialized dictionaries for Pydantic models and `Annotated[dict[str, TaskResult], merge_task_results]` for task results. Include `message`, `run_id`, `triage`, `plan`, `expected_task_ids`, `task_results`, `rag_results`, `draft`, `draft_version`, `proposed_actions`, `review_request`, `review_decision`, `revision_count`, `events`, `delivery`, `case_id`, `error`, and `outcome`.

- [ ] **Step 5: Define narrow ports and the reducer**

```python
# src/pwc_support/ports/knowledge.py
from typing import Protocol
from pwc_support.domain.models import RagRequest, RagResult, RetrievalHit


class KnowledgePort(Protocol):
    def retrieve(self, request: RagRequest, *, limit: int = 6) -> tuple[RetrievalHit, ...]: ...
    def answer(self, request: RagRequest) -> RagResult: ...
```

```python
# src/pwc_support/workflow/reducers.py
from pwc_support.domain.models import TaskResult


def merge_task_results(
    left: dict[str, TaskResult] | None,
    right: dict[str, TaskResult] | None,
) -> dict[str, TaskResult]:
    merged = dict(left or {})
    for task_id, result in (right or {}).items():
        existing = merged.get(task_id)
        if existing is not None and existing != result:
            raise ValueError(f"conflicting task result: {task_id}")
        merged[task_id] = result
    return merged
```

Define `LLMPort.classify`, `plan`, `compose`, and `verify`; `CasePort.lookup`, `create`, and `update`; and `MailboxPort.receive`, `reply`, `conversation`, and `delivery_status`, all using Task 2 models and no framework-specific types.

- [ ] **Step 6: Run domain tests and static checks**

Run:

```bash
uv run pytest tests/unit/domain tests/unit/workflow/test_reducers.py -q
uv run ruff check src tests
uv run mypy src tests
```

Expected: all domain and reducer tests pass; static checks exit `0`.

- [ ] **Step 7: Commit domain contracts**

```bash
git add src/pwc_support/domain src/pwc_support/ports src/pwc_support/workflow tests/unit
git commit -m "feat: define support workflow contracts"
```

### Task 3: SQLite Operations Store and Idempotent Case Tools

**Files:**
- Create: `src/pwc_support/adapters/sqlite/schema.sql`
- Create: `src/pwc_support/adapters/sqlite/database.py`
- Create: `src/pwc_support/adapters/sqlite/repositories.py`
- Create: `tests/unit/adapters/test_sqlite_repositories.py`
- Create: `tests/contract/test_case_port.py`

**Interfaces:**
- Consumes: `IncomingMessage`, case/review/outbox models, and `CasePort` from Task 2.
- Produces: `SQLiteDatabase.initialize()`, `OperationsRepository`, and `SQLiteCaseAdapter` implementing `CasePort`.

- [ ] **Step 1: Write failing transaction, access, and idempotency tests**

```python
def test_case_creation_is_idempotent(repository, case_request) -> None:
    first = repository.create_case(case_request, idempotency_key="create:run-1")
    second = repository.create_case(case_request, idempotency_key="create:run-1")
    assert second == first
    assert repository.count_cases() == 1


def test_client_cannot_read_another_clients_case(repository, seeded_case) -> None:
    with pytest.raises(SupportError) as caught:
        repository.lookup_case(seeded_case.case_id, requester_id="different-client")
    assert caught.value.code is ErrorCode.CASE_ACCESS_DENIED


def test_outbox_rejects_second_body_for_same_idempotency_key(repository, outgoing_reply) -> None:
    repository.write_outbox(outgoing_reply, idempotency_key="reply:conversation-1:v1")
    changed = outgoing_reply.model_copy(update={"body": "different"})
    with pytest.raises(SupportError, match="idempotency conflict"):
        repository.write_outbox(changed, idempotency_key="reply:conversation-1:v1")
```

- [ ] **Step 2: Run repository tests and verify missing adapter**

Run: `uv run pytest tests/unit/adapters/test_sqlite_repositories.py -q`

Expected: FAIL because `pwc_support.adapters.sqlite` is missing.

- [ ] **Step 3: Create schema version 1**

Create `schema.sql` with `PRAGMA foreign_keys = ON`, a `schema_versions` table, and the seven tables from the spec. Use text UUIDs, ISO-8601 UTC timestamps, JSON text columns validated in Python, `UNIQUE(provider, provider_message_id)`, `UNIQUE(provider, provider_thread_id, client_id)` for conversations, `UNIQUE(idempotency_key)` for cases/outbox operations, and foreign keys with explicit delete behavior. Create indexes on pending review status, case client ID, conversation timestamp, and workflow run status.

The migration transaction must insert version `1` only after every table and index succeeds.

- [ ] **Step 4: Implement the connection factory**

```python
class SQLiteDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        sql = resources.files("pwc_support.adapters.sqlite").joinpath("schema.sql").read_text()
        with self.connect() as connection:
            connection.executescript(sql)
```

- [ ] **Step 5: Implement repositories and case adapter**

Implement transaction-scoped methods for conversation/message upsert, duplicate detection by content hash, workflow-run start/finish, case lookup/create/update, review create/list/decide, and outbox write/read. Generate case IDs transactionally from an integer sequence and format them as `DEMO-{number:03d}` only after the insert succeeds.

Use optimistic locking for updates:

```sql
UPDATE cases
SET status = ?, summary = ?, version = version + 1, updated_at = ?
WHERE case_id = ? AND version = ?
```

If `rowcount != 1`, raise `SupportError(ErrorCode.VERSION_CONFLICT, ...)`. On repeated idempotency keys, deserialize and return the existing identical result; raise an idempotency conflict when the stored request hash differs.

- [ ] **Step 6: Run repository and port contract tests**

Run:

```bash
uv run pytest tests/unit/adapters/test_sqlite_repositories.py tests/contract/test_case_port.py -q
uv run ruff check src tests
uv run mypy src tests
```

Expected: persistence, access, optimistic-locking, and idempotency tests pass.

- [ ] **Step 7: Commit operational persistence**

```bash
git add src/pwc_support/adapters/sqlite tests/unit/adapters tests/contract
git commit -m "feat: add idempotent SQLite case storage"
```

### Task 4: Review Policy and Sanitized Operational Events

**Files:**
- Create: `src/pwc_support/workflow/policy.py`
- Create: `src/pwc_support/observability/events.py`
- Create: `tests/unit/workflow/test_policy.py`
- Create: `tests/unit/observability/test_events.py`

**Interfaces:**
- Consumes: `ReviewCategory`, `IncomingMessage`, and `OperationalEvent` from Task 2.
- Produces: `ReviewPolicy.evaluate(text) -> PolicyDecision` and `EventRecorder.record(event) -> None`.

- [ ] **Step 1: Write failing review-policy tests**

```python
@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("A shared link exposed confidential documents", ReviewCategory.CONFIDENTIALITY),
        ("Is this transaction compliant with the regulation?", ReviewCategory.LEGAL_REGULATORY),
        ("I want to make a formal complaint", ReviewCategory.COMPLAINT_ESCALATION),
        ("Please schedule a meeting and commit to Friday", ReviewCategory.EXTERNAL_ACTION),
        ("Give a definitive audit conclusion for our engagement", ReviewCategory.PROFESSIONAL_JUDGEMENT),
    ],
)
def test_mandatory_categories_are_detected(text: str, category: ReviewCategory) -> None:
    decision = ReviewPolicy.default().evaluate(text)
    assert category in decision.categories
    assert decision.must_review is True


def test_industry_alone_does_not_force_review() -> None:
    decision = ReviewPolicy.default().evaluate("What services are available to insurers?")
    assert decision.must_review is False
```

- [ ] **Step 2: Write the failing log-redaction test**

```python
def test_event_recorder_does_not_write_message_or_email(tmp_path: Path) -> None:
    destination = tmp_path / "events.jsonl"
    recorder = EventRecorder(destination)
    recorder.record(
        OperationalEvent(
            run_id=uuid4(), conversation_id=uuid4(), node="triage",
            event_type="completed", details={"message": "secret", "email": "a@example.com"},
        )
    )
    written = destination.read_text()
    assert "secret" not in written
    assert "a@example.com" not in written
    assert "redacted_fields" in written
```

- [ ] **Step 3: Run tests and verify missing modules**

Run: `uv run pytest tests/unit/workflow/test_policy.py tests/unit/observability/test_events.py -q`

Expected: collection fails because policy and event modules do not exist.

- [ ] **Step 4: Implement deterministic policy evaluation**

```python
@dataclass(frozen=True)
class PolicyDecision:
    must_review: bool
    categories: frozenset[ReviewCategory]
    matched_rule_ids: tuple[str, ...]


@dataclass(frozen=True)
class ReviewRule:
    rule_id: str
    category: ReviewCategory
    pattern: re.Pattern[str]


class ReviewPolicy:
    def __init__(self, rules: tuple[ReviewRule, ...]) -> None:
        self._rules = rules

    @classmethod
    def default(cls) -> "ReviewPolicy":
        definitions = (
            ("confidential-incident", ReviewCategory.CONFIDENTIALITY,
             r"\b(confidential|privacy|data breach|cyber(?:security)?|expos(?:e|ed|ure)|leak(?:ed)?)\b"),
            ("legal-regulatory", ReviewCategory.LEGAL_REGULATORY,
             r"\b(legal advice|regulat(?:ion|ory)|compliance conclusion|lawful|legally)\b"),
            ("complaint-escalation", ReviewCategory.COMPLAINT_ESCALATION,
             r"\b(formal complaint|escalat(?:e|ion)|speak to (?:a )?(?:person|manager|human))\b"),
            ("external-action", ReviewCategory.EXTERNAL_ACTION,
             r"\b(schedule|book|send on my behalf|commit|sign|approve|submit)\b"),
            ("professional-judgement", ReviewCategory.PROFESSIONAL_JUDGEMENT,
             r"\b(definitive|engagement-specific|audit conclusion|tax advice|assurance opinion)\b"),
        )
        return cls(tuple(ReviewRule(rule_id, category, re.compile(pattern, re.I))
                         for rule_id, category, pattern in definitions))

    def evaluate(self, text: str) -> PolicyDecision:
        matches = tuple(rule for rule in self._rules if rule.pattern.search(text))
        categories = frozenset(rule.category for rule in matches)
        return PolicyDecision(bool(categories), categories, tuple(rule.rule_id for rule in matches))
```

Evidence insufficiency and conflict are inputs to `ReviewPolicy.after_retrieval(...)`, not keyword rules. That method adds `INSUFFICIENT_EVIDENCE` whenever RAG returns insufficient/conflicting evidence.

- [ ] **Step 5: Implement JSONL event recording with an allowlist**

Allow only `route`, `status`, `duration_ms`, `model`, `attempt`, `task_count`, `hit_count`, `citation_count`, `error_code`, and numeric token fields in persisted `details`. Record the names of dropped keys in `redacted_fields`. Create the destination directory, append one sorted JSON object per line, flush each write, and protect writes with a lock.

- [ ] **Step 6: Run policy, observability, and static checks**

Run:

```bash
uv run pytest tests/unit/workflow/test_policy.py tests/unit/observability/test_events.py -q
uv run ruff check src tests
uv run mypy src tests
```

Expected: all tests pass, and no test event file contains private fields.

- [ ] **Step 7: Commit policy and events**

```bash
git add src/pwc_support/workflow/policy.py src/pwc_support/observability tests/unit
git commit -m "feat: enforce review policy and safe events"
```

### Task 5: Ollama Generation and Embedding Adapter

**Files:**
- Create: `src/pwc_support/adapters/ollama.py`
- Create: `tests/unit/adapters/test_ollama_adapter.py`
- Create: `tests/integration/test_ollama_runtime.py`
- Create: `scripts/check_runtime.py`

**Interfaces:**
- Consumes: `Settings`, LLM port models, and installed Ollama models.
- Produces: `OllamaAdapter` implementing `LLMPort`; `embed_documents`, `embed_query`, and `health()`.

- [ ] **Step 1: Write failing unit tests with a stub Ollama client**

```python
class StubClient:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.chat_calls: list[dict[str, object]] = []
        self.embed_calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> dict[str, object]:
        self.chat_calls.append(kwargs)
        return self.responses.pop(0)

    def embed(self, **kwargs: object) -> dict[str, object]:
        self.embed_calls.append(kwargs)
        return {"embeddings": [[0.1, 0.2, 0.3]]}


def test_query_and_document_prefixes_are_distinct(settings) -> None:
    client = StubClient([])
    adapter = OllamaAdapter(settings, client=client)
    adapter.embed_documents(["document text"])
    adapter.embed_query("question text")
    assert client.embed_calls[0]["input"] == ["search_document: document text"]
    assert client.embed_calls[1]["input"] == ["search_query: question text"]


def test_invalid_schema_gets_one_repair_attempt(settings, valid_triage_json) -> None:
    client = StubClient([
        {"message": {"content": "not json"}},
        {"message": {"content": valid_triage_json}},
    ])
    decision = OllamaAdapter(settings, client=client).classify("What services do you offer?")
    assert decision.route is Route.PLAN
    assert len(client.chat_calls) == 2
```

- [ ] **Step 2: Run tests and verify the missing adapter**

Run: `uv run pytest tests/unit/adapters/test_ollama_adapter.py -q`

Expected: FAIL because `OllamaAdapter` is not defined.

- [ ] **Step 3: Implement bounded structured generation**

Construct `ollama.Client(host=settings.ollama_base_url, timeout=settings.request_timeout_seconds)`. Guard every `chat` call with `threading.BoundedSemaphore(settings.max_parallel_generations)`. Pass each Pydantic schema through `format=Model.model_json_schema()`, use `options={"num_ctx": ..., "num_predict": ..., "temperature": ...}`, and parse only `response["message"]["content"]`.

```python
def _structured_chat[T: BaseModel](
    self,
    *,
    schema: type[T],
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> T:
    for attempt in range(2):
        prompt = user if attempt == 0 else f"Return only valid JSON for this schema.\n{user}"
        with self._generation_slots:
            response = self._client.chat(
                model=self._settings.generation_model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                format=schema.model_json_schema(),
                options={"num_ctx": self._settings.num_ctx,
                         "num_predict": max_tokens, "temperature": temperature},
            )
        try:
            return schema.model_validate_json(str(response["message"]["content"]))
        except (KeyError, TypeError, ValidationError) as error:
            if attempt == 1:
                raise SupportError(ErrorCode.MODEL_OUTPUT_INVALID, str(error)) from error
    raise AssertionError("unreachable")
```

Map timeout/connection failures to stable `SupportError` codes. Do not call `ollama.pull()`.

- [ ] **Step 4: Implement embeddings and health checks**

Batch documents according to settings, validate a constant nonzero dimension within each result, and L2-normalize only when the returned vectors are not already unit length within `1e-4`. `health()` calls `list()` and `show()` for both configured models and returns model names, digests, dimensions where available, and Ollama version. Missing models cause a diagnostic error containing the explicit `ollama pull` command; they are never pulled automatically.

- [ ] **Step 5: Add runtime check script and opt-in integration test**

```python
# scripts/check_runtime.py
from pwc_support.adapters.ollama import OllamaAdapter
from pwc_support.config import Settings


def main() -> int:
    report = OllamaAdapter(Settings.from_env()).health()
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Mark the integration test with `@pytest.mark.integration`; assert one query embedding, one document embedding, matching dimensions, and a valid structured triage decision. Do not assert exact generated prose.

- [ ] **Step 6: Run unit gates, then run the explicit runtime probe**

Run:

```bash
uv run pytest tests/unit/adapters/test_ollama_adapter.py -q
uv run ruff check src tests scripts
uv run mypy src tests scripts
uv run python scripts/check_runtime.py
uv run pytest tests/integration/test_ollama_runtime.py -q
```

Expected: unit gates pass. Runtime commands either pass with model metadata or stop with an exact missing/unreachable diagnostic; resolve that environment issue before marking Task 5 complete.

- [ ] **Step 7: Commit the Ollama adapter**

```bash
git add src/pwc_support/adapters/ollama.py tests scripts/check_runtime.py
git commit -m "feat: add bounded local Ollama adapter"
```

### Task 6: Curated Corpus Manifest and Deterministic Chunking

**Files:**
- Create: `corpus/manifest.json`
- Create: `corpus/documents/pwc-global-services.md`
- Create: `corpus/documents/pwc-financial-services.md`
- Create: `corpus/documents/pwc-industries.md`
- Create: `corpus/documents/pwc-network-structure.md`
- Create: `corpus/documents/synthetic-support-faq.md`
- Create: `src/pwc_support/rag/chunking.py`
- Create: `src/pwc_support/rag/ingestion.py`
- Create: `tests/unit/rag/test_chunking.py`
- Create: `tests/unit/rag/test_manifest.py`

**Interfaces:**
- Consumes: local Markdown and `OllamaAdapter.embed_documents`.
- Produces: `load_manifest(path) -> CorpusManifest`, `chunk_document(document) -> tuple[Chunk, ...]`, and `prepare_chunks(manifest) -> tuple[Chunk, ...]`.

- [ ] **Step 1: Write failing manifest and chunking tests**

```python
def test_chunk_ids_are_stable_and_chunks_do_not_cross_h1() -> None:
    document = SourceDocument(
        source_id="source-1", title="Title", text="# A\n" + "a " * 500 + "\n# B\n" + "b " * 90,
        language="en", source_type="public_summary", canonical_url="https://example.com",
        metadata={"source_status": "active"},
    )
    first = chunk_document(document, max_words=350, overlap_words=50)
    second = chunk_document(document, max_words=350, overlap_words=50)
    assert first == second
    assert all(not ("a a" in chunk.text and "b b" in chunk.text) for chunk in first)
    assert max(len(chunk.text.split()) for chunk in first) <= 350


def test_manifest_fails_when_checksum_does_not_match(tmp_path: Path) -> None:
    manifest_path = write_manifest_with_wrong_checksum(tmp_path)
    with pytest.raises(CorpusValidationError, match="checksum"):
        load_manifest(manifest_path)
```

- [ ] **Step 2: Run tests and verify missing RAG modules**

Run: `uv run pytest tests/unit/rag/test_chunking.py tests/unit/rag/test_manifest.py -q`

Expected: collection fails because chunking and ingestion modules are missing.

- [ ] **Step 3: Add the five reviewed source documents**

Write concise, paraphrased summaries with visible headings and inline canonical-source attribution. The four public summaries may state only the findings already documented in the spec. `synthetic-support-faq.md` must begin with:

```markdown
# Synthetic support process fixtures

This document is fictional test data for the prototype. It is not a PwC policy,
service commitment, response-time promise, or description of an internal process.
```

It may define only demo wording for receiving a case, requesting territory, and indicating pending review. Do not insert actual names, email addresses, engagement data, prices, or promised response times.

- [ ] **Step 4: Build and validate the manifest**

Create one JSON object per source with `source_id`, `title`, `local_path`, `canonical_url`, `accessed_at: "2026-09-02"`, `language: "en"`, `sector`, `service_line`, `territory_scope`, `source_type`, `source_status: "active"`, and a SHA-256 checksum computed from exact UTF-8 bytes. Public entries use `source_type: "public_summary"`; the FAQ uses `synthetic_fixture` and a null canonical URL.

`load_manifest` must reject duplicate IDs, paths outside `corpus/documents`, unknown metadata keys, unsupported language, non-active status values, absent files, and checksum mismatches.

- [ ] **Step 5: Implement heading-aware chunking**

Split on Markdown headings, preserve heading paths, normalize whitespace, make windows of at most 350 words with 50-word overlap, and merge sub-80-word trailing chunks into the previous chunk only when the result remains within 350 words. Derive IDs as:

```python
payload = "\x1f".join((source_id, heading_path, str(chunk_index), text, "chunker-v1"))
chunk_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

Return source metadata with every chunk and never place source instructions into prompt roles.

- [ ] **Step 6: Run corpus tests and inspect deterministic output**

Run:

```bash
uv run pytest tests/unit/rag/test_chunking.py tests/unit/rag/test_manifest.py -q
uv run python -c "from pathlib import Path; from pwc_support.rag.ingestion import load_manifest, prepare_chunks; m=load_manifest(Path('corpus/manifest.json')); c=prepare_chunks(m); print(len(m.sources), len(c), c[0].chunk_id)"
```

Expected: tests pass, five sources are reported, every chunk has a stable ID, and no validation warning appears.

- [ ] **Step 7: Commit corpus processing**

```bash
git add corpus src/pwc_support/rag tests/unit/rag
git commit -m "feat: add validated support corpus processing"
```

### Task 7: Chroma Storage and Idempotent Ingestion CLI

**Files:**
- Create: `src/pwc_support/adapters/chroma.py`
- Create: `scripts/ingest_corpus.py`
- Create: `tests/contract/test_knowledge_store.py`
- Create: `tests/unit/adapters/test_chroma_adapter.py`
- Create: `tests/integration/test_chroma_ingestion.py`

**Interfaces:**
- Consumes: Task 6 chunks and Task 5 embeddings.
- Produces: `ChromaStore.upsert_chunks`, `delete_stale`, `retrieve`, `collection_metadata`, and an idempotent ingestion command.

- [ ] **Step 1: Write failing Chroma adapter tests**

```python
def test_ingestion_is_idempotent(chroma_store, sample_chunks) -> None:
    first = chroma_store.sync_source(sample_chunks[0].source_id, sample_chunks)
    second = chroma_store.sync_source(sample_chunks[0].source_id, sample_chunks)
    assert first.inserted == len(sample_chunks)
    assert second.inserted == 0
    assert second.unchanged == len(sample_chunks)
    assert chroma_store.count() == len(sample_chunks)


def test_retrieval_falls_back_when_specific_filter_is_too_narrow(chroma_store) -> None:
    result = chroma_store.retrieve(
        RagRequest(question="insurance services", language="en", sector="insurance"),
        limit=6,
    )
    assert result.used_filter_fallback is True
    assert len(result.hits) >= 2
```

- [ ] **Step 2: Run tests and verify missing Chroma adapter**

Run: `uv run pytest tests/unit/adapters/test_chroma_adapter.py -q`

Expected: FAIL because `ChromaStore` is not defined.

- [ ] **Step 3: Implement configurable Chroma clients**

```python
def build_chroma_client(settings: Settings) -> chromadb.ClientAPI:
    if settings.chroma_mode == "persistent":
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(settings.chroma_path))
    return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
```

Create/get `settings.collection_name` with cosine distance and collection metadata containing embedding tag/digest, dimension, chunker version, and corpus checksum. Reject startup when existing metadata conflicts. Convert cosine distance to similarity with `1.0 - distance` and preserve the raw distance for diagnostics.

- [ ] **Step 4: Implement synchronized upsert and retrieval**

For each source, fetch existing chunk IDs, upsert changed/new chunks in embedding batches of 16, delete stale IDs, and return inserted/updated/unchanged/deleted counts. Metadata includes only Chroma-compatible scalar values; lists are encoded as stable pipe-delimited strings.

Retrieve with `language=en` and `source_status=active`. Add exact sector/service/territory conditions only when the request carries a triage confidence of at least `0.80`. When fewer than two eligible hits remain, run one broad fallback and mark `used_filter_fallback=True`. Never pass user-provided operators directly into Chroma's `where` object.

- [ ] **Step 5: Add the ingestion CLI**

```python
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("corpus/manifest.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    manifest = load_manifest(args.manifest)
    chunks = prepare_chunks(manifest)
    report = IngestionService(settings).sync(manifest, chunks, dry_run=args.dry_run)
    print(report.model_dump_json(indent=2))
    return 0
```

Dry-run validates and computes hashes/counts without embedding or mutating Chroma.

- [ ] **Step 6: Run unit/contract checks, then isolated integration ingestion**

Run:

```bash
uv run pytest tests/unit/adapters/test_chroma_adapter.py tests/contract/test_knowledge_store.py -q
uv run python scripts/ingest_corpus.py --dry-run
PWC_DATA_DIR=/tmp/pwc-support-integration uv run python scripts/ingest_corpus.py
PWC_DATA_DIR=/tmp/pwc-support-integration uv run pytest tests/integration/test_chroma_ingestion.py -q
```

Expected: dry-run reports five validated sources without an embedding call; two real syncs produce identical collection counts and no duplicates.

- [ ] **Step 7: Commit vector storage and ingestion**

```bash
git add src/pwc_support/adapters/chroma.py scripts/ingest_corpus.py tests
git commit -m "feat: index and retrieve support knowledge"
```

### Task 8: Independently Callable Four-Node RAG Subgraph

**Files:**
- Create: `config/retrieval.json`
- Create: `src/pwc_support/rag/graph.py`
- Create: `tests/unit/rag/test_rag_graph.py`
- Create: `tests/integration/test_rag_graph_real.py`

**Interfaces:**
- Consumes: `RagRequest`, `ChromaStore.retrieve`, `OllamaAdapter.embed_query`, and `LLMPort.answer`.
- Produces: `build_rag_graph(dependencies) -> CompiledStateGraph` whose input contains `request` and whose output contains a validated `RagResult`.

- [ ] **Step 1: Write failing graph-route tests using fakes**

```python
def test_rag_returns_cited_answer_when_evidence_is_sufficient(rag_graph, rag_request) -> None:
    result = rag_graph.invoke({"request": rag_request.model_dump(mode="json")})
    rag_result = RagResult.model_validate(result["result"])
    assert rag_result.status is RagStatus.ANSWERED
    assert {citation.source_id for citation in rag_result.citations} == {"pwc-financial-services"}
    assert result["visited_nodes"] == [
        "prepare_query", "retrieve", "select_evidence", "answer_with_citations"
    ]


def test_rag_abstains_without_calling_answer_model_when_below_threshold(
    insufficient_rag_graph,
    rag_request,
) -> None:
    result = insufficient_rag_graph.invoke({"request": rag_request.model_dump(mode="json")})
    assert result["result"]["status"] == "insufficient_evidence"
    assert insufficient_rag_graph.fake_llm.answer_calls == 0
```

- [ ] **Step 2: Run tests and verify the graph is missing**

Run: `uv run pytest tests/unit/rag/test_rag_graph.py -q`

Expected: FAIL because `build_rag_graph` is not defined.

- [ ] **Step 3: Add exact retrieval configuration**

```json
{
  "schema_version": 1,
  "top_k": 6,
  "max_selected_chunks": 4,
  "max_evidence_characters": 6000,
  "min_similarity": 0.45,
  "min_specific_filter_hits": 2,
  "specific_filter_min_confidence": 0.8,
  "threshold_candidates": [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
}
```

Validate this file with a Pydantic `RetrievalConfig` and reject unknown fields or unsorted threshold candidates.

- [ ] **Step 4: Implement RAG state and four nodes**

```python
class RagState(TypedDict, total=False):
    request: dict[str, object]
    prepared_query: str
    retrieval: dict[str, object]
    selected_hits: list[dict[str, object]]
    result: dict[str, object]
    visited_nodes: Annotated[list[str], operator.add]
```

`prepare_query` adds only the query prefix once. `retrieve` asks the knowledge store for six hits. `select_evidence` removes duplicate chunk IDs, keeps active English hits at or above the configured threshold, limits per-source repetition to two chunks, and stops at four chunks or 6,000 characters. It returns an insufficient result when no eligible evidence remains or the retrieval batch flags a conflict.

`answer_with_citations` passes numbered evidence blocks such as `<source id="S1" chunk="...">...</source>` as untrusted text. It validates model-returned markers, joins canonical metadata from selected hits, rejects unknown markers, and emits no source supplied only by the model.

- [ ] **Step 5: Compile and expose the subgraph**

```python
def build_rag_graph(dependencies: RagDependencies) -> CompiledStateGraph:
    builder = StateGraph(RagState)
    builder.add_node("prepare_query", dependencies.prepare_query)
    builder.add_node("retrieve", dependencies.retrieve)
    builder.add_node("select_evidence", dependencies.select_evidence)
    builder.add_node("answer_with_citations", dependencies.answer_with_citations)
    builder.add_edge(START, "prepare_query")
    builder.add_edge("prepare_query", "retrieve")
    builder.add_edge("retrieve", "select_evidence")
    builder.add_conditional_edges(
        "select_evidence",
        route_selected_evidence,
        {"answer": "answer_with_citations", "insufficient": END},
    )
    builder.add_edge("answer_with_citations", END)
    return builder.compile()
```

Keep subgraph persistence per invocation; parent workflow checkpointing persists the returned result.

- [ ] **Step 6: Run fake and real RAG verification**

Run:

```bash
uv run pytest tests/unit/rag/test_rag_graph.py -q
uv run pytest tests/integration/test_rag_graph_real.py -q
uv run ruff check src tests
uv run mypy src tests
```

Expected: fake tests prove both branches and exact node visitation; the real test returns either a valid cited answer or an explicit insufficient-evidence result without invented citation IDs.

- [ ] **Step 7: Commit the RAG subgraph**

```bash
git add config/retrieval.json src/pwc_support/rag/graph.py tests
git commit -m "feat: add modular cited RAG subgraph"
```

### Task 9: Intake, Triage, Planning, and Independent Task Dispatch

**Files:**
- Create: `src/pwc_support/workflow/nodes.py`
- Create: `src/pwc_support/workflow/routing.py`
- Create: `tests/unit/workflow/test_intake_triage.py`
- Create: `tests/unit/workflow/test_planning_dispatch.py`

**Interfaces:**
- Consumes: domain models, review policy, repositories, LLM port, RAG graph, and case port.
- Produces: implementations of main nodes 1–4 and routing functions that return node names or LangGraph `Send` objects.

- [ ] **Step 1: Write failing intake and triage tests**

```python
def test_intake_deduplicates_provider_message(intake_node, received_email) -> None:
    first = intake_node({"message": received_email.model_dump(mode="json")})
    second = intake_node({"message": received_email.model_dump(mode="json")})
    assert first["duplicate"] is False
    assert second["duplicate"] is True
    assert first["run_id"] == second["run_id"]


def test_deterministic_review_match_cannot_be_removed_by_llm(triage_node, incident_state) -> None:
    update = triage_node(incident_state)
    decision = TriageDecision.model_validate(update["triage"])
    assert decision.route is Route.REVIEW
    assert ReviewCategory.CONFIDENTIALITY in decision.review_categories
```

- [ ] **Step 2: Write the failing independent-dispatch test**

```python
def test_compound_plan_dispatches_knowledge_and_case_tasks(compound_state) -> None:
    sends = dispatch_planned_tasks(compound_state)
    assert [send.node for send in sends] == ["rag_subgraph", "case_tools"]
    assert {send.arg["task"]["kind"] for send in sends} == {
        "knowledge_query", "case_lookup"
    }
```

- [ ] **Step 3: Run tests and verify missing workflow nodes**

Run: `uv run pytest tests/unit/workflow/test_intake_triage.py tests/unit/workflow/test_planning_dispatch.py -q`

Expected: FAIL because node and routing functions do not exist.

- [ ] **Step 4: Implement `intake` and `triage`**

`intake` validates `IncomingMessage`, hashes normalized content, calls the mailbox receive operation, starts one workflow-run record, and returns the existing run for exact duplicates. It rejects unsupported language before any model call.

`triage` runs deterministic policy first. If a mandatory rule matches, build `TriageDecision(route=REVIEW, ...)` without asking the model to weaken it. Otherwise call `LLMPort.classify`, union any model categories with deterministic categories, normalize sector/service values against fixed enums, and use `CLARIFY` when territory or another required field is explicitly missing. Invalid output or model failure returns a review decision with an error reason.

- [ ] **Step 5: Implement bounded planning**

For a single public-information intent, create one deterministic `knowledge_query` task. For a single case status request with a syntactically valid `DEMO-[0-9]{3,}` ID, create one `case_lookup` task. For compound input, call `LLMPort.plan`, reject dependency cycles, reject unknown task kinds, cap tasks at four, and convert every external action into a `ProposedAction` that forces review.

Task IDs are UUIDv5 values derived from run ID, task kind, and normalized task input so retries create the same IDs.

- [ ] **Step 6: Implement task routing and node 4**

```python
def dispatch_planned_tasks(state: SupportState) -> list[Send] | str:
    plan = WorkPlan.model_validate(state["plan"])
    if not plan.tasks:
        return "compose_reply"
    targets = {
        TaskKind.KNOWLEDGE_QUERY: "rag_subgraph",
        TaskKind.CASE_LOOKUP: "case_tools",
    }
    return [Send(targets[task.kind], {"task": task.model_dump(mode="json")})
            for task in plan.tasks]
```

`case_tools` validates the task, calls only `CasePort.lookup`, catches stable domain errors into failed `TaskResult`, and returns `{task_results: {task_id: result}}`. RAG task adaptation similarly turns the subgraph result into one keyed `TaskResult`.

- [ ] **Step 7: Run workflow-front tests and static gates**

Run:

```bash
uv run pytest tests/unit/workflow/test_intake_triage.py tests/unit/workflow/test_planning_dispatch.py -q
uv run ruff check src tests
uv run mypy src tests
```

Expected: mandatory routes bypass unsafe model weakening, compound plans dispatch both independent task types, and all tests pass.

- [ ] **Step 8: Commit main workflow front half**

```bash
git add src/pwc_support/workflow tests/unit/workflow
git commit -m "feat: triage and decompose support enquiries"
```

### Task 10: Composition, Verification, Human Review, and Finalization

**Files:**
- Modify: `src/pwc_support/workflow/nodes.py`
- Modify: `src/pwc_support/workflow/routing.py`
- Create: `tests/unit/workflow/test_composition_verification.py`
- Create: `tests/unit/workflow/test_review_finalization.py`

**Interfaces:**
- Consumes: terminal task results from Task 9, review repository, mailbox, and case ports.
- Produces: main nodes 5–8; deterministic release gates; JSON-serializable interrupt packets; idempotent final client outcomes.

- [ ] **Step 1: Write failing composition-barrier and citation tests**

```python
def test_composition_waits_for_every_expected_task(compose_node, state_with_one_of_two_results) -> None:
    with pytest.raises(SupportError, match="task barrier"):
        compose_node(state_with_one_of_two_results)


def test_unknown_citation_forces_review(verify_node, drafted_state) -> None:
    drafted_state["draft"]["citation_markers"] = ["S999"]
    update = verify_node(drafted_state)
    assert update["verification"]["route"] == "review"
    assert update["verification"]["error_code"] == "EVIDENCE_INSUFFICIENT"
```

- [ ] **Step 2: Write failing interrupt/version/idempotency tests**

```python
def test_old_review_version_is_rejected(review_node, pending_state, monkeypatch) -> None:
    monkeypatch.setattr(nodes, "interrupt", lambda packet: {
        "decision": "approve", "response_version": pending_state["draft_version"] - 1,
        "reviewer_id": "reviewer-1",
    })
    with pytest.raises(SupportError) as caught:
        review_node(pending_state)
    assert caught.value.code is ErrorCode.VERSION_CONFLICT


def test_finalization_retry_writes_one_reply(finalise_node, approved_state, repository) -> None:
    first = finalise_node(approved_state)
    second = finalise_node(approved_state)
    assert second["outcome"] == first["outcome"]
    assert repository.count_outbox() == 1
```

- [ ] **Step 3: Run tests and verify missing back-half behavior**

Run: `uv run pytest tests/unit/workflow/test_composition_verification.py tests/unit/workflow/test_review_finalization.py -q`

Expected: FAIL because composition, review, and finalization are not implemented.

- [ ] **Step 4: Implement composition and clarification**

Check `set(expected_task_ids) == set(task_results)` and that each result is terminal. A missing-context route creates one focused clarification without calling RAG. A single successful knowledge result may reuse its cited answer. Compound results call `LLMPort.compose` with bounded successful payloads and explicit failed-task notices. Any failed required task yields clarification, review, or unable-to-answer according to its stable error code.

The draft stores text, marker list, source IDs, task IDs, and an integer version incremented on every material change.

- [ ] **Step 5: Implement deterministic verification**

```python
def verify_citations(draft: DraftReply, hits: tuple[RetrievalHit, ...]) -> Verification:
    allowed = {f"S{index}": hit for index, hit in enumerate(hits, start=1)}
    markers = set(draft.citation_markers)
    unknown = markers - allowed.keys()
    if unknown or (draft.factual and not markers):
        return Verification.review(ErrorCode.EVIDENCE_INSUFFICIENT)
    return Verification.release(citations=tuple(allowed[marker] for marker in sorted(markers)))
```

Add policy categories, evidence conflict, external action approval, 1,500-character reply length, response-version match, and revision count. The semantic `LLMPort.verify` may change `release` to `review` or `revise`; it cannot change a deterministic review to release. A second revision failure routes to review.

- [ ] **Step 6: Implement durable human review**

Persist `ReviewRequest` before `interrupt(packet)`. The packet includes only JSON-serializable case ID, original message summary, categories, citations, proposed reply, actions, and current version. On resume, validate `ReviewDecision`; approve/edit routes back through verification, request-revision increments the revision count and routes to composition, and reject/take-ownership routes to finalization without an automated substantive reply. Update the review record idempotently.

- [ ] **Step 7: Implement finalization**

Derive idempotency keys as `case:{run_id}:{action_version}` and `reply:{conversation_id}:{draft_version}`. Create/update a case only for a declared action or review outcome. Join a newly created case ID into a controlled acknowledgement after successful creation. Write one simulated reply for an automatic verified answer or approved reviewer response. For rejection or ownership without reply text, record the administrative outcome and write no outbox row.

- [ ] **Step 8: Run back-half tests and static gates**

Run:

```bash
uv run pytest tests/unit/workflow/test_composition_verification.py tests/unit/workflow/test_review_finalization.py -q
uv run ruff check src tests
uv run mypy src tests
```

Expected: barrier, citation, version, resume, rejection, and duplicate-finalization tests pass.

- [ ] **Step 9: Commit main workflow back half**

```bash
git add src/pwc_support/workflow tests/unit/workflow
git commit -m "feat: verify pause and finalize support replies"
```

### Task 11: Assemble the Eight-Node Graph and Application Services

**Files:**
- Create: `src/pwc_support/workflow/graph.py`
- Create: `src/pwc_support/adapters/simulated_mailbox.py`
- Create: `src/pwc_support/services/client_support.py`
- Create: `src/pwc_support/services/review.py`
- Create: `src/pwc_support/bootstrap.py`
- Create: `scripts/seed_demo.py`
- Create: `tests/contract/test_mailbox_port.py`
- Create: `tests/unit/services/test_client_support.py`
- Create: `tests/integration/test_workflow_end_to_end.py`

**Interfaces:**
- Consumes: every adapter and node from Tasks 3–10.
- Produces: `build_main_graph(dependencies, checkpointer)`, `ApplicationContainer`, `ClientSupportService`, `ReviewService`, and reproducible demo records.

- [ ] **Step 1: Write failing graph-shape and end-to-end fake tests**

```python
EXPECTED_MAIN_NODES = {
    "intake", "triage", "plan_work", "case_tools",
    "compose_reply", "verify_response", "human_review", "finalise_case",
}


def test_main_graph_has_exactly_eight_counted_nodes(fake_container) -> None:
    graph = fake_container.graph.get_graph()
    counted = set(graph.nodes) - {"__start__", "__end__", "rag_subgraph"}
    assert counted == EXPECTED_MAIN_NODES


def test_general_question_is_automatically_released(fake_service, general_message) -> None:
    outcome = fake_service.submit(general_message)
    assert outcome.status is OutcomeStatus.ANSWERED
    assert outcome.citations
    assert fake_service.repository.count_outbox() == 1


def test_incident_pauses_and_resumes_same_conversation(
    fake_service, fake_review_service, incident_message
) -> None:
    pending = fake_service.submit(incident_message)
    assert pending.status is OutcomeStatus.PENDING_REVIEW
    resumed = fake_review_service.decide(
        pending.review_id,
        ReviewDecision.approve(reviewer_id="reviewer-1", response_version=1),
    )
    assert resumed.conversation_id == incident_message.conversation_id
```

- [ ] **Step 2: Run tests and verify missing assembly**

Run: `uv run pytest tests/unit/services/test_client_support.py tests/integration/test_workflow_end_to_end.py -q`

Expected: FAIL because graph assembly and services are missing.

- [ ] **Step 3: Compile the exact graph with SQLite checkpointing**

Register exactly the eight named main nodes plus the separately compiled `rag_subgraph`. Add conditional routing from triage, plan work, verification, and human review exactly as specified. Use `SqliteSaver(sqlite3.connect(path, check_same_thread=False))`, set WAL/busy timeout, and retain the connection for the application lifecycle. Invoke every run with:

```python
config = {"configurable": {"thread_id": str(message.conversation_id)}}
```

Use LangGraph v2 result/interrupt access consistently and add a graph-shape test that fails if counted node names change.

- [ ] **Step 4: Implement mailbox adapter and client service**

`SimulatedMailboxAdapter.receive` delegates deduplication to the operations repository. `reply` derives the recipient and thread from the stored conversation and never accepts a model-generated destination. `ClientSupportService.submit` starts/continues the graph, maps interrupts to `PENDING_REVIEW`, maps final graph state to `ClientOutcome`, and returns the stored outcome for duplicate provider messages.

`get_conversation` verifies client ID and renders messages plus status; `get_run_events` returns only the whitelisted operational projection.

- [ ] **Step 5: Implement review service and bootstrap container**

`ReviewService.list_pending` reads persisted packets. `decide` checks reviewer ID, packet status, and response version, then calls the graph with `Command(resume=decision.model_dump(mode="json"))` and the original conversation thread ID.

`ApplicationContainer.build(settings)` creates directories, initializes schema, creates event recorder, Ollama/Chroma/case/mailbox adapters, retrieval config, RAG graph, checkpoint saver, main graph, and services. Add `close()` for checkpoint connections.

- [ ] **Step 6: Add deterministic demo seeding**

Seed clients `northstar-insurance` and `harbour-financial`, authorized case `DEMO-104`, one general email question, and one confidentiality incident. Use fixed UUIDs and timestamps so repeated seeding is idempotent. Label every name and record as synthetic.

- [ ] **Step 7: Run service, contract, and graph tests**

Run:

```bash
uv run pytest tests/contract/test_mailbox_port.py tests/unit/services -q
uv run pytest tests/integration/test_workflow_end_to_end.py -q
uv run python scripts/seed_demo.py
uv run python scripts/seed_demo.py
```

Expected: exact graph shape passes; general answer releases automatically; critical flow pauses/resumes; the second seed changes no row counts.

- [ ] **Step 8: Commit graph assembly and services**

```bash
git add src/pwc_support scripts/seed_demo.py tests
git commit -m "feat: assemble persistent support application"
```

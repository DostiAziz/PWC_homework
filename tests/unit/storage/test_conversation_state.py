from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from pwc_support.domain.models import ConversationMemory, SpecialistName
from pwc_support.storage.conversation_state import ConversationStateRepository
from pwc_support.storage.database import Database


def _memory(client_id: str = "CUS-1") -> ConversationMemory:
    return ConversationMemory(
        conversation_id=uuid4(),
        client_id=client_id,
        active_specialist=SpecialistName.ORDER,
        active_task_id="order-1",
        state_json={"phase": "awaiting_order_id", "attempts": 0},
        version=0,
        updated_at=datetime.now(UTC),
    )


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
        updated_at=datetime.now(UTC),
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


def test_oversized_state_payload_is_rejected(tmp_path: Path) -> None:
    database = Database(tmp_path / "support.sqlite")
    repository = ConversationStateRepository(database, max_state_chars=100)
    oversized = _memory().model_copy(update={"state_json": {"blob": "x" * 200}})
    with pytest.raises(ValueError, match="conversation memory exceeds"):
        repository.save(oversized, expected_version=None)


def test_clear_removes_the_stored_memory(tmp_path: Path) -> None:
    database = Database(tmp_path / "support.sqlite")
    repository = ConversationStateRepository(database)
    saved = repository.save(_memory(), expected_version=None)

    repository.clear(saved.conversation_id, saved.client_id, expected_version=saved.version)

    assert repository.load(saved.conversation_id, saved.client_id) is None


def test_clear_rejects_stale_version(tmp_path: Path) -> None:
    database = Database(tmp_path / "support.sqlite")
    repository = ConversationStateRepository(database)
    saved = repository.save(_memory(), expected_version=None)

    with pytest.raises(ValueError, match="memory version conflict"):
        repository.clear(saved.conversation_id, saved.client_id, expected_version=saved.version + 1)

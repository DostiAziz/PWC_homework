from pathlib import Path
from typing import Any

import bootstrap
from config import Settings


class FakeDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        pass


class FakeKnowledgeBase:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.collection = object()

    def count(self) -> int:
        return 1


def test_runtime_wires_generation_budget_and_parallel_limit(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_chat_model(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(bootstrap, "Database", FakeDatabase)
    monkeypatch.setattr(bootstrap, "get_embeddings", lambda **kwargs: object())
    monkeypatch.setattr(bootstrap, "get_chat_model", fake_chat_model)
    monkeypatch.setattr(bootstrap, "embedding_dimension", lambda embedder: 384)
    monkeypatch.setattr(
        bootstrap, "validate_collection_dimension", lambda collection, dimension, **kwargs: None
    )
    monkeypatch.setattr(bootstrap, "chroma_client", lambda settings: object())
    monkeypatch.setattr(bootstrap, "ChromaKnowledgeBase", FakeKnowledgeBase)
    monkeypatch.setattr(bootstrap, "LexicalIndex", lambda path: object())
    monkeypatch.setattr(bootstrap, "RagAnswerer", lambda *args, **kwargs: object())
    monkeypatch.setattr(bootstrap, "ToolRegistry", lambda *args: object())
    monkeypatch.setattr(bootstrap, "ProductRepository", lambda database: object())
    monkeypatch.setattr(bootstrap, "OrderRepository", lambda database: object())
    monkeypatch.setattr(bootstrap, "build_agent_graph", lambda **kwargs: object())
    monkeypatch.setattr(bootstrap, "AgentService", lambda graph: object())

    bootstrap.build_runtime(
        Settings(answer_tokens=91, max_parallel_generations=2, retail_db_path=Path("fake.db"))
    )

    assert captured["max_output_tokens"] == 91
    assert captured["max_parallel_generations"] == 2

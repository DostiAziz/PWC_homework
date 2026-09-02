from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, HttpUrl


class CorpusDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    path: str
    title: str
    url: HttpUrl | None = None
    language: str = "en"
    source_status: str = "active"
    text: str = ""


def load_manifest(path: Path) -> tuple[CorpusDocument, ...]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return tuple(CorpusDocument.model_validate(item) for item in payload["documents"])


def load_documents(root: Path, manifest: tuple[CorpusDocument, ...]) -> tuple[CorpusDocument, ...]:
    return tuple(
        document.model_copy(update={"text": (root / document.path).read_text(encoding="utf-8")})
        for document in manifest
    )

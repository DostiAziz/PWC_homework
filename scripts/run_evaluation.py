"""Score the frozen evaluation set against the real Ollama, Chroma, and SQLite runtime."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bootstrap import build_runtime
from config import Settings
from domain.models import ChatReply
from storage.database import Database

try:
    from scripts.seed_retail_data import seed
except ModuleNotFoundError:
    from seed_retail_data import seed  # type: ignore[import-not-found,no-redef]

from pydantic import BaseModel, ConfigDict, Field

CRITERIA = ("routing", "sources", "terms", "safety", "attribution")
MARKER = re.compile(r"\[S\d+\]")


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    turns: list[str] = Field(default_factory=list)
    customer_id: str = "CUS-1001"
    expected_tools: list[str] = Field(default_factory=list)
    allowed_tools: list[str] | None = None
    forbidden_tools: list[str] = Field(default_factory=list)
    required_sources: list[str] = Field(default_factory=list)
    required_terms: list[str] = Field(default_factory=list)
    forbidden_terms: list[str] = Field(default_factory=list)
    expect_citations: bool = False
    expect_confirmation: bool | None = None
    expected_orders: dict[str, dict[str, Any]] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CaseScore:
    case_id: str
    passed: bool
    checks: dict[str, bool]
    detail: dict[str, Any]


def score_case(
    case: dict[str, Any] | EvaluationCase,
    reply: ChatReply,
    db_passed: bool = True,
) -> CaseScore:
    case_data = case.model_dump() if isinstance(case, BaseModel) else dict(case)
    text = (
        reply.message.casefold()
        .replace("\u2011", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    expected_tools = set(case_data.get("expected_tools", []))
    actual_steps = set(reply.steps)
    forbidden_tools = set(case_data.get("forbidden_tools", []))
    allowed_tools = set(case_data["allowed_tools"]) if case_data.get("allowed_tools") else None

    required_sources = set(case_data.get("required_sources", []))
    actual_sources = {citation.source_id for citation in reply.citations}
    message_markers = set(MARKER.findall(reply.message))
    citation_markers = {citation.marker for citation in reply.citations}
    expect_citations = bool(case_data.get("expect_citations", False))

    routing_ok = expected_tools <= actual_steps
    if allowed_tools is not None:
        routing_ok = routing_ok and (actual_steps <= allowed_tools)
    if forbidden_tools:
        routing_ok = routing_ok and not bool(actual_steps & forbidden_tools)

    has_forbidden_terms = any(
        term.casefold() in text for term in case_data.get("forbidden_terms", [])
    )
    confirmation_ok = True
    if "expect_confirmation" in case_data and case_data["expect_confirmation"] is not None:
        confirmation_ok = reply.awaiting_confirmation == bool(case_data["expect_confirmation"])

    checks = {
        "routing": routing_ok,
        "sources": required_sources <= actual_sources,
        "terms": all(term.casefold() in text for term in case_data.get("required_terms", [])),
        "safety": not has_forbidden_terms and confirmation_ok and db_passed,
        "attribution": (
            bool(reply.citations) == expect_citations
            and message_markers <= citation_markers
            and (not expect_citations or bool(message_markers))
        ),
    }
    return CaseScore(
        case_id=str(case_data["id"]),
        passed=all(checks.values()),
        checks=checks,
        detail={
            "answer": reply.message,
            "steps": list(reply.steps),
            "cited_sources": sorted(actual_sources),
            "latency_ms": reply.total_duration_ms,
        },
    )


def run(path: Path) -> dict[str, Any]:
    settings = Settings.from_env()
    raw_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = [EvaluationCase.model_validate(json.loads(line)) for line in raw_lines]
    scores: list[CaseScore] = []
    started = time.perf_counter()

    for index, case in enumerate(cases, start=1):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / f"eval_{case.id}.sqlite3"
            database = Database(db_path)
            database.initialize()
            seed(database)

            eval_settings = settings.model_copy(update={"retail_db_path": db_path})
            runtime = build_runtime(eval_settings)
            service = runtime.service

            customer_id = case.customer_id
            thread_id = str(uuid.uuid4())
            reply: ChatReply | None = None
            turns = case.turns if case.turns else [case.id]

            for turn in turns:
                if reply is not None and reply.awaiting_confirmation:
                    reply = service.resume(
                        thread_id=thread_id,
                        customer_id=customer_id,
                        decision=str(turn),
                    )
                else:
                    reply = service.submit(
                        thread_id=thread_id, body=str(turn), customer_id=customer_id
                    )
            assert reply is not None

            # Verify DB expectations if requested
            db_passed = True
            if case.expected_orders:
                with database.connect() as conn:
                    for order_id, expected_fields in case.expected_orders.items():
                        row = conn.execute(
                            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
                        ).fetchone()
                        if row is None:
                            db_passed = False
                            break
                        for col, expected_val in expected_fields.items():
                            if row[col] != expected_val:
                                db_passed = False
                                break

            score = score_case(case, reply, db_passed=db_passed)
            scores.append(score)
            print(
                f"[{index:>2}/{len(cases)}] {score.case_id:<28} "
                f"{'PASS' if score.passed else 'FAIL'} "
                f"({score.detail['latency_ms']:.0f} ms)"
            )

    elapsed = time.perf_counter() - started
    per_criterion = {
        criterion: round(sum(score.checks[criterion] for score in scores) / len(scores), 4)
        if scores
        else 0.0
        for criterion in CRITERIA
    }
    return {
        "generation_model": settings.generation_model,
        "embedding_model": settings.embedding_model,
        "cases": len(scores),
        "passed": sum(score.passed for score in scores),
        "accuracy": round(sum(score.passed for score in scores) / len(scores), 4)
        if scores
        else 0.0,
        "per_criterion_accuracy": per_criterion,
        "elapsed_seconds": round(elapsed, 2),
        "results": [
            {
                "id": score.case_id,
                "passed": score.passed,
                "checks": score.checks,
                **score.detail,
            }
            for score in scores
        ],
    }


def main() -> None:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("eval/final.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=settings.artifacts_dir / "evaluation" / "final-result.json"
    )
    args = parser.parse_args()
    result = run(args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"\nAccuracy {result['accuracy']:.1%} over {result['cases']} cases "
        f"in {result['elapsed_seconds']:.1f}s -> {args.output}"
    )
    for criterion, value in result["per_criterion_accuracy"].items():
        print(f"  {criterion:<14} {value:.1%}")


if __name__ == "__main__":
    main()

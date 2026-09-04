"""Score the frozen evaluation set against the real Ollama, Chroma, and SQLite runtime."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pwc_support.bootstrap import build_runtime
from pwc_support.config import Settings
from pwc_support.domain.models import ChatReply
from pwc_support.storage.database import Database

try:
    from scripts.seed_retail_data import seed
except ModuleNotFoundError:
    from seed_retail_data import seed  # type: ignore[import-not-found,no-redef]

CRITERIA = ("routing", "sources", "terms", "safety", "attribution", "conversation")
MARKER = re.compile(r"\[S\d+\]")


@dataclass(frozen=True, slots=True)
class CaseScore:
    case_id: str
    passed: bool
    checks: dict[str, bool]
    detail: dict[str, Any]


def score_case(case: dict[str, Any], reply: ChatReply) -> CaseScore:
    text = reply.message.casefold()
    expected_kinds = sorted(case.get("expected_task_kinds", []))
    actual_kinds = sorted(task.kind.value for task in reply.tasks)
    required_sources = set(case.get("required_sources", []))
    actual_sources = {citation.source_id for citation in reply.citations}
    message_markers = set(MARKER.findall(reply.message))
    citation_markers = {citation.marker for citation in reply.citations}
    expect_citations = bool(case.get("expect_citations", False))
    checks = {
        "routing": actual_kinds == expected_kinds,
        "sources": required_sources <= actual_sources,
        "terms": all(term.casefold() in text for term in case.get("required_terms", [])),
        "safety": all(term.casefold() not in text for term in case.get("forbidden_terms", [])),
        "attribution": (
            bool(reply.citations) == expect_citations
            and message_markers <= citation_markers
            and (not expect_citations or bool(message_markers))
        ),
        "conversation": (
            (reply.pending_cancellation is not None)
            == bool(case.get("expect_pending_cancellation", False))
        ),
    }
    return CaseScore(
        case_id=str(case["id"]),
        passed=all(checks.values()),
        checks=checks,
        detail={
            "answer": reply.message,
            "task_kinds": actual_kinds,
            "cited_sources": sorted(actual_sources),
            "latency_ms": reply.total_duration_ms,
        },
    )


def run(path: Path) -> dict[str, Any]:
    settings = Settings.from_env()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "eval_retail.sqlite3"
        database = Database(db_path)
        database.initialize()
        seed(database)

        eval_settings = settings.model_copy(update={"retail_db_path": db_path})
        runtime = build_runtime(eval_settings)
        service = runtime.service

        cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        scores: list[CaseScore] = []
        started = time.perf_counter()
        for index, case in enumerate(cases, start=1):
            customer_id = str(case.get("customer_id", "CUS-1001"))
            pending_cancellation = None
            reply: ChatReply | None = None
            turns = case.get("turns", [case.get("question", "")])
            for turn in turns:
                reply = service.submit(
                    body=str(turn),
                    customer_id=customer_id,
                    pending_cancellation=pending_cancellation,
                )
                pending_cancellation = reply.pending_cancellation
            assert reply is not None
            score = score_case(case, reply)
            scores.append(score)
            print(
                f"[{index:>2}/{len(cases)}] {score.case_id:<24} "
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
            "generation_model": runtime.settings.generation_model,
            "embedding_model": runtime.settings.embedding_model,
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

"""Score the frozen evaluation set against the real Ollama, Chroma and SQLite runtime."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from pwc_support.bootstrap import build_runtime
from pwc_support.services.client_support import ClientSupportService, WorkflowRun

CRITERIA = ("status", "sources", "abstention", "categories", "tasks", "attribution")
MARKER = re.compile(r"\[S\d+\]")


@dataclass(frozen=True, slots=True)
class CaseScore:
    case_id: str
    passed: bool
    checks: dict[str, bool]
    detail: dict[str, Any]


def score_case(case: dict[str, Any], run: WorkflowRun) -> CaseScore:
    """Score route, cited sources, abstention, review categories, decomposition, attribution."""
    status = run.outcome.status.value
    expected_status = str(case["expected_status"])
    cited = sorted({citation.source_id for citation in run.outcome.citations})
    required = sorted(case.get("required_sources", []))
    categories = sorted(run.state.get("triage", {}).get("review_categories", []))
    task_kinds = sorted(
        str(task["kind"]) for task in run.state.get("plan", {}).get("tasks", [])
    )
    expected_kinds = sorted(case.get("expected_task_kinds", []))
    answered = status == "answered" and bool(run.outcome.citations)
    checks = {
        "status": status == expected_status,
        "sources": set(required) <= set(cited),
        # An unsupported question must abstain rather than invent an answer.
        "abstention": expected_status != "unable_to_answer" or not run.outcome.citations,
        "categories": set(case.get("expected_categories", [])) <= set(categories),
        "tasks": task_kinds == expected_kinds,
        # A released answer must carry at least one marker that maps to a real citation.
        "attribution": not answered
        or bool(
            set(MARKER.findall(run.outcome.message))
            & {citation.marker for citation in run.outcome.citations}
        ),
    }
    return CaseScore(
        case_id=str(case["id"]),
        passed=all(checks.values()),
        checks=checks,
        detail={
            "question": case["question"],
            "expected_status": expected_status,
            "actual_status": status,
            "cited_sources": cited,
            "required_sources": required,
            "review_categories": categories,
            "task_kinds": task_kinds,
            "latency_ms": run.total_duration_ms,
            "visited_nodes": run.visited_nodes,
            "answer": run.outcome.message[:400],
        },
    )


def run(path: Path) -> dict[str, Any]:
    runtime = build_runtime(checkpointer=InMemorySaver(), enable_interrupt=True)
    service = ClientSupportService(
        runtime.graph,
        reviews=runtime.reviews,
        database=runtime.database,
        mailbox=runtime.mailbox,
    )
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    scores: list[CaseScore] = []
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        result = service.submit(
            body=str(case["question"]),
            client_id=f"evaluation-client-{index}",
        )
        score = score_case(case, result)
        scores.append(score)
        print(
            f"[{index:>2}/{len(cases)}] {score.case_id:<24} "
            f"{'PASS' if score.passed else 'FAIL'} "
            f"({score.detail['actual_status']}, {score.detail['latency_ms']:.0f} ms)"
        )
    elapsed = time.perf_counter() - started
    per_criterion = {
        criterion: round(
            sum(score.checks[criterion] for score in scores) / len(scores), 4
        )
        for criterion in CRITERIA
    }
    return {
        "generation_model": runtime.settings.generation_model,
        "embedding_model": runtime.settings.embedding_model,
        "cases": len(scores),
        "passed": sum(score.passed for score in scores),
        "accuracy": round(sum(score.passed for score in scores) / len(scores), 4),
        "per_criterion_accuracy": per_criterion,
        "elapsed_seconds": round(elapsed, 2),
        "results": [
            {"id": score.case_id, "passed": score.passed, "checks": score.checks,
             **score.detail}
            for score in scores
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("eval/final.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/evaluation/final-result.json")
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
        print(f"  {criterion:<12} {value:.1%}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pwc_support.services.client_support import ClientSupportService
from pwc_support.workflow.graph import build_graph


def run(path: Path) -> dict[str, Any]:
    service = ClientSupportService(build_graph())
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    results = []
    for case in cases:
        outcome = service.submit(body=case["question"], client_id="evaluation-client")
        expected = "answered" if case["expected_route"] == "answer" else "pending_review"
        results.append({"id": case["id"], "expected": expected, "actual": outcome.status.value,
                        "passed": outcome.status.value == expected})
    return {"cases": results, "accuracy": sum(item["passed"] for item in results) / len(results)}


if __name__ == "__main__":
    result = run(Path("eval/final.jsonl"))
    output = Path("artifacts/evaluation/final-result.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

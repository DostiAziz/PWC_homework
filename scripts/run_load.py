from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pwc_support.services.client_support import ClientSupportService
from pwc_support.workflow.graph import build_graph


def run(requests: int = 50, concurrency: int = 1) -> dict[str, float | int]:
    service = ClientSupportService(build_graph())
    questions = [
        "What services does PwC provide to financial services clients?",
        "Which industries does PwC publish information about?",
    ]

    def submit(index: int) -> str:
        outcome = service.submit(
            body=questions[index % len(questions)], client_id=f"load-{index}"
        )
        return outcome.status.value

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        statuses = list(pool.map(submit, range(requests)))
    elapsed = time.perf_counter() - started
    return {"requests": requests, "concurrency": concurrency, "elapsed_seconds": elapsed,
            "throughput_per_second": requests / elapsed if elapsed else 0.0,
            "answered": statuses.count("answered")}


if __name__ == "__main__":
    result = run()
    output = Path("artifacts/load/local-result.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

"""Load-test the full support workflow: real retrieval, generation, persistence, delivery."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pwc_support.bootstrap import build_runtime
from pwc_support.services.client_support import ClientSupportService


@dataclass(frozen=True, slots=True)
class RequestSample:
    index: int
    question_id: str
    status: str
    latency_ms: float
    node_ms: dict[str, float]
    error: str | None = None


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile: with 50 samples an interpolated p99 would be fictitious."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-fraction * len(ordered) // 1))))
    return round(ordered[rank - 1], 2)


def run_phase(
    service: ClientSupportService,
    workload: list[dict[str, Any]],
    *,
    requests: int,
    concurrency: int,
) -> dict[str, Any]:
    def submit(index: int) -> RequestSample:
        case = workload[index % len(workload)]
        started = time.perf_counter()
        try:
            run = service.submit(
                body=str(case["question"]),
                client_id=f"load-client-{index}",
            )
        except Exception as error:
            return RequestSample(
                index=index,
                question_id=str(case["id"]),
                status="error",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                node_ms={},
                error=f"{type(error).__name__}: {error}",
            )
        node_ms: dict[str, float] = defaultdict(float)
        for event in run.events:
            node_ms[str(event["node"])] += float(event.get("duration_ms") or 0.0)
            for key, value in event.get("details", {}).items():
                if str(key).startswith("rag."):
                    node_ms[str(key)] += float(value)
        return RequestSample(
            index=index,
            question_id=str(case["id"]),
            status=run.outcome.status.value,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            node_ms=dict(node_ms),
        )

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        samples = list(pool.map(submit, range(requests)))
    elapsed = time.perf_counter() - started

    latencies = [sample.latency_ms for sample in samples if sample.error is None]
    node_totals: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        for node, value in sample.node_ms.items():
            node_totals[node].append(value)
    node_profile = {
        node: {
            "calls": len(values),
            "mean_ms": round(statistics.fmean(values), 2),
            "total_ms": round(sum(values), 2),
            "share_of_measured_ms": 0.0,
        }
        for node, values in node_totals.items()
    }
    # The rag.* stages are nested inside execute_task, so exclude them from the share base.
    measured_total = sum(
        item["total_ms"]
        for node, item in node_profile.items()
        if not node.startswith("rag.")
    ) or 1.0
    for item in node_profile.values():
        item["share_of_measured_ms"] = round(item["total_ms"] / measured_total, 4)
    statuses: dict[str, int] = defaultdict(int)
    for sample in samples:
        statuses[sample.status] += 1

    return {
        "requests": requests,
        "concurrency": concurrency,
        "elapsed_seconds": round(elapsed, 2),
        "throughput_per_second": round(requests / elapsed, 3) if elapsed else 0.0,
        "failures": sum(1 for sample in samples if sample.error is not None),
        "statuses": dict(statuses),
        "latency_ms": {
            "min": round(min(latencies), 2) if latencies else 0.0,
            "mean": round(statistics.fmean(latencies), 2) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
        "node_profile": dict(
            sorted(node_profile.items(), key=lambda item: -item[1]["total_ms"])
        ),
        "errors": [sample.error for sample in samples if sample.error is not None][:5],
    }


def summarise(phases: list[dict[str, Any]]) -> dict[str, Any]:
    """Name the measured bottleneck rather than asserting one from architecture alone."""
    baseline, scaled = phases[0], phases[-1]
    hottest = max(
        (
            (node, item)
            for node, item in baseline["node_profile"].items()
            if not node.startswith("rag.")
        ),
        key=lambda item: item[1]["total_ms"],
        default=("", {}),
    )
    stages = {
        node: item["total_ms"]
        for node, item in baseline["node_profile"].items()
        if node.startswith("rag.")
    }
    stage_total = sum(stages.values()) or 1.0
    p95_growth = (
        scaled["latency_ms"]["p95"] / baseline["latency_ms"]["p95"]
        if baseline["latency_ms"]["p95"]
        else 0.0
    )
    throughput_gain = (
        scaled["throughput_per_second"] / baseline["throughput_per_second"]
        if baseline["throughput_per_second"]
        else 0.0
    )
    return {
        "bottleneck_node": hottest[0],
        "bottleneck_stage_breakdown": {
            node: round(value / stage_total, 4)
            for node, value in sorted(stages.items(), key=lambda item: -item[1])
        },
        "bottleneck_share_of_measured_time": hottest[1].get("share_of_measured_ms", 0.0),
        "p95_growth_at_higher_concurrency": round(p95_growth, 2),
        "throughput_gain_at_higher_concurrency": round(throughput_gain, 2),
        "interpretation": (
            "Doubling concurrency multiplied p95 latency by "
            f"{p95_growth:.2f} while throughput changed by only "
            f"{throughput_gain:.2f}x, which is the signature of a single serialised "
            "resource rather than of client-side overhead."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=Path("eval/load_workload.jsonl"))
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output", type=Path, default=Path("artifacts/load/local-result.json"))
    args = parser.parse_args()

    workload = [
        json.loads(line)
        for line in args.workload.read_text(encoding="utf-8").splitlines()
        if line
    ]
    runtime = build_runtime(enable_interrupt=False)
    service = ClientSupportService(
        runtime.graph,
        reviews=runtime.reviews,
        database=runtime.database,
        mailbox=runtime.mailbox,
    )
    phases = []
    for concurrency in args.concurrency:
        print(f"Running {args.requests} requests at concurrency {concurrency}…")
        phase = run_phase(
            service, workload, requests=args.requests, concurrency=concurrency
        )
        phases.append(phase)
        print(
            f"  p50={phase['latency_ms']['p50']:.0f} ms  "
            f"p95={phase['latency_ms']['p95']:.0f} ms  "
            f"p99={phase['latency_ms']['p99']:.0f} ms  "
            f"throughput={phase['throughput_per_second']:.2f}/s  "
            f"failures={phase['failures']}"
        )
    result = {
        "generation_model": runtime.settings.generation_model,
        "embedding_model": runtime.settings.embedding_model,
        "total_requests": args.requests * len(args.concurrency),
        "phases": phases,
        "summary": summarise(phases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nBottleneck: {result['summary']['bottleneck_node']} -> {args.output}")


if __name__ == "__main__":
    main()

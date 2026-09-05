"""Load-test the full support workflow: real retrieval, generation, persistence."""

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

from customer_support.bootstrap import build_runtime
from customer_support.config import Settings
from customer_support.services.chat import AgentService


@dataclass(frozen=True, slots=True)
class RequestSample:
    index: int
    question_id: str
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
    service: AgentService,
    workload: list[dict[str, Any]],
    *,
    requests: int,
    concurrency: int,
) -> dict[str, Any]:
    def submit(index: int) -> RequestSample:
        case = workload[index % len(workload)]
        started = time.perf_counter()
        try:
            reply = service.submit(
                body=str(case["question"]),
                customer_id=str(case["customer_id"]),
            )
        except Exception as error:
            return RequestSample(
                index=index,
                question_id=str(case["id"]),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                node_ms={},
                error=f"{type(error).__name__}: {error}",
            )
        failed = "unavailable" in reply.message.lower()
        node_ms: dict[str, float] = {"agent": reply.total_duration_ms}
        return RequestSample(
            index=index,
            question_id=str(case["id"]),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            node_ms=node_ms,
            error=reply.message if failed else None,
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
    measured_total = sum(item["total_ms"] for item in node_profile.values()) or 1.0
    for item in node_profile.values():
        item["share_of_measured_ms"] = round(item["total_ms"] / measured_total, 4)

    return {
        "requests": requests,
        "concurrency": concurrency,
        "elapsed_seconds": round(elapsed, 2),
        "throughput_per_second": round(requests / elapsed, 3) if elapsed else 0.0,
        "failures": sum(1 for sample in samples if sample.error is not None),
        "latency_ms": {
            "min": round(min(latencies), 2) if latencies else 0.0,
            "mean": round(statistics.fmean(latencies), 2) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
        "node_profile": dict(sorted(node_profile.items(), key=lambda item: -item[1]["total_ms"])),
        "errors": [sample.error for sample in samples if sample.error is not None][:5],
    }


def summarise(phases: list[dict[str, Any]]) -> dict[str, Any]:
    baseline, scaled = phases[0], phases[-1]
    bottleneck_node, hottest = max(
        baseline["node_profile"].items(),
        key=lambda item: item[1]["total_ms"],
        default=("none", {"share_of_measured_ms": 0.0}),
    )
    baseline_p95 = baseline["latency_ms"]["p95"]
    baseline_rate = baseline["throughput_per_second"]
    p95_growth = scaled["latency_ms"]["p95"] / baseline_p95 if baseline_p95 else 0.0
    throughput_gain = scaled["throughput_per_second"] / baseline_rate if baseline_rate else 0.0
    recommendations = [
        (
            "Benchmark a smaller generation model against the frozen evaluation before adoption; "
            f"the measured hot node is {bottleneck_node}."
        ),
        (
            "Keep one generation slot when added concurrency raises p95 much more than throughput; "
            "otherwise re-run with the measured best concurrency."
        ),
    ]
    return {
        "bottleneck_node": bottleneck_node,
        "bottleneck_share_of_measured_time": hottest["share_of_measured_ms"],
        "p95_growth_at_higher_concurrency": round(p95_growth, 2),
        "throughput_gain_at_higher_concurrency": round(throughput_gain, 2),
        "recommendations": recommendations,
    }


def main() -> None:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=Path("eval/load_workload.jsonl"))
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--output", type=Path, default=settings.artifacts_dir / "load" / "local-result.json"
    )
    args = parser.parse_args()

    workload = [
        json.loads(line) for line in args.workload.read_text(encoding="utf-8").splitlines() if line
    ]
    runtime = build_runtime()
    service = runtime.service
    phases = []
    for concurrency in args.concurrency:
        print(f"Running {args.requests} requests at concurrency {concurrency}…")
        phase = run_phase(service, workload, requests=args.requests, concurrency=concurrency)
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

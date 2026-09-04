from scripts.run_load import percentile, summarise


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.95) == 40.0


def test_summary_names_measured_hottest_node() -> None:
    phases = [
        {
            "concurrency": 1,
            "throughput_per_second": 1.0,
            "latency_ms": {"p95": 100.0},
            "node_profile": {
                "plan_tasks": {"total_ms": 800.0, "share_of_measured_ms": 0.8},
                "catalogue_task": {"total_ms": 200.0, "share_of_measured_ms": 0.2},
            },
        },
        {
            "concurrency": 2,
            "throughput_per_second": 1.1,
            "latency_ms": {"p95": 180.0},
            "node_profile": {},
        },
    ]

    summary = summarise(phases)

    assert summary["bottleneck_node"] == "plan_tasks"
    assert summary["p95_growth_at_higher_concurrency"] == 1.8
    assert len(summary["recommendations"]) == 2

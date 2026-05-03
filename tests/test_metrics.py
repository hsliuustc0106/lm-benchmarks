import json
from unittest.mock import patch
import lm_benchmarks.metrics as metrics


SAMPLE_VLLM_OUTPUT = {
    "mean_ttft_ms": 450.2,
    "median_ttft_ms": 380.5,
    "p99_ttft_ms": 1200.0,
    "mean_tpot_ms": 25.3,
    "median_tpot_ms": 24.1,
    "p99_tpot_ms": 55.0,
    "request_throughput": 7.8,
    "output_throughput": 4200.0,
    "total_token_throughput": 38000.0,
}


def test_collect_reads_vllm_json(tmp_path):
    """collect() reads vllm output JSON and extracts metrics."""
    result_file = tmp_path / "result.json"
    with open(result_file, "w") as f:
        json.dump(SAMPLE_VLLM_OUTPUT, f)

    run_config = {"model": "test/model", "request_rate": 8.0, "max_concurrency": 32}
    gpu_samples = [
        {"timestamp": 0, "utilization_pct": 85, "memory_used_mb": 40000},
        {"timestamp": 1, "utilization_pct": 90, "memory_used_mb": 41000},
    ]

    run_cfg_path, metrics_path = metrics.collect(
        result_json=result_file,
        run_config=run_config,
        gpu_samples=gpu_samples,
        output_dir=tmp_path,
    )

    assert run_cfg_path.exists()
    assert metrics_path.exists()

    with open(metrics_path) as f:
        m = json.load(f)
        assert m["mean_ttft_ms"] == 450.2
        assert m["mean_tpot_ms"] == 25.3
        assert m["output_throughput"] == 4200.0
        assert m["gpu_avg_utilization_pct"] == 87.5


def test_collect_computes_tokens_per_user(tmp_path):
    """metrics includes tokens_per_user = output_throughput / concurrency."""
    result_file = tmp_path / "result.json"
    with open(result_file, "w") as f:
        json.dump({"output_throughput": 4000.0, "request_throughput": 8.0}, f)

    run_config = {"max_concurrency": 16}
    _, metrics_path = metrics.collect(
        result_file, run_config, [], output_dir=tmp_path
    )

    with open(metrics_path) as f:
        m = json.load(f)
    assert m["tokens_per_user"] == 250.0  # 4000 / 16


def test_collect_handles_missing_vllm_fields(tmp_path):
    """Missing fields default to None, don't crash."""
    result_file = tmp_path / "result.json"
    with open(result_file, "w") as f:
        json.dump({}, f)

    _, metrics_path = metrics.collect(
        result_file, {}, [], output_dir=tmp_path
    )

    with open(metrics_path) as f:
        m = json.load(f)
    assert m["mean_ttft_ms"] is None


def test_run_config_includes_timestamps(tmp_path):
    """Written run_config includes started_at and completed_at."""
    result_file = tmp_path / "result.json"
    with open(result_file, "w") as f:
        json.dump({}, f)

    cfg_path, _ = metrics.collect(
        result_file, {"model": "x"}, [], output_dir=tmp_path
    )

    with open(cfg_path) as f:
        cfg = json.load(f)
    assert "run_id" in cfg
    assert "started_at" in cfg
    assert "completed_at" in cfg
    assert cfg["model"] == "x"


def test_gpu_metrics_empty_when_no_samples(tmp_path):
    """GPU stats are None when no samples provided."""
    result_file = tmp_path / "result.json"
    with open(result_file, "w") as f:
        json.dump(SAMPLE_VLLM_OUTPUT, f)

    _, metrics_path = metrics.collect(
        result_file, {}, [], output_dir=tmp_path
    )

    with open(metrics_path) as f:
        m = json.load(f)
    assert m["gpu_avg_utilization_pct"] is None


def test_compute_percentiles():
    """_compute_percentiles returns p50, p90, p95, p99."""
    data = list(range(1, 101))  # 1..100
    result = metrics._compute_percentiles(data)
    assert result["p50"] == 50.5  # median of 1..100
    assert result["p95"] == 95.05
    assert "p90" in result
    assert "p99" in result


def test_summary_stats():
    """_summary_stats returns mean, min, max, count."""
    result = metrics._summary_stats([10, 20, 30])
    assert result["mean"] == 20.0
    assert result["min"] == 10.0
    assert result["max"] == 30.0
    assert result["count"] == 3

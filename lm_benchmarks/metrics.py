"""Metrics collection: read vllm output, compute statistics, write results."""
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lm_benchmarks.utils import generate_run_id, load_json, save_json, utc_timestamp


def collect(
    result_json: Path,
    run_config: Dict[str, Any],
    gpu_samples: List[Dict[str, Any]],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Process vllm benchmark output into run_config.json and run_metrics.json.

    Returns (run_config_path, run_metrics_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_json(result_json) or {}

    # Build run_config with resolved parameters and metadata
    cfg = dict(run_config)
    cfg["run_id"] = generate_run_id()
    cfg["started_at"] = utc_timestamp()
    cfg["completed_at"] = utc_timestamp()
    cfg["result_file"] = str(result_json)

    cfg_path = output_dir / "run_config.json"
    save_json(cfg_path, cfg)

    # Build metrics
    concurrency = run_config.get("max_concurrency", 1) or 1
    output_throughput = raw.get("output_throughput")

    gpu_utils = [s["utilization_pct"] for s in gpu_samples if "utilization_pct" in s]

    metrics_data = {
        "run_id": cfg["run_id"],
        "model": run_config.get("model"),
        "request_rate": run_config.get("request_rate"),
        "max_concurrency": run_config.get("max_concurrency"),
        "mean_ttft_ms": raw.get("mean_ttft_ms"),
        "median_ttft_ms": raw.get("median_ttft_ms"),
        "p99_ttft_ms": raw.get("p99_ttft_ms"),
        "mean_tpot_ms": raw.get("mean_tpot_ms"),
        "median_tpot_ms": raw.get("median_tpot_ms"),
        "p99_tpot_ms": raw.get("p99_tpot_ms"),
        "request_throughput": raw.get("request_throughput"),
        "output_throughput": output_throughput,
        "total_token_throughput": raw.get("total_token_throughput"),
        "tokens_per_user": output_throughput / concurrency if output_throughput else None,
        "gpu_avg_utilization_pct": statistics.mean(gpu_utils) if gpu_utils else None,
        "gpu_max_utilization_pct": max(gpu_utils) if gpu_utils else None,
    }

    metrics_path = output_dir / "run_metrics.json"
    save_json(metrics_path, metrics_data)

    return cfg_path, metrics_path


def _compute_percentiles(values: List[float]) -> Dict[str, float]:
    """Compute p50, p90, p95, p99 from a list of values."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def pct(p):
        k = (n - 1) * p / 100
        f = int(k)
        c = k - f
        if f + 1 < n:
            return sorted_vals[f] + c * (sorted_vals[f + 1] - sorted_vals[f])
        return sorted_vals[f]

    return {
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
    }


def _summary_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean, min, max, count."""
    if not values:
        return {"mean": 0, "min": 0, "max": 0, "count": 0}
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }

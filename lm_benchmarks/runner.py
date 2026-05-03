"""Benchmark orchestration: sweep, single run, GPU sampling."""
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lm_benchmarks import metrics, plot, serve, utils


def _build_bench_cmd(
    port: int,
    request_rate: float,
    max_concurrency: int,
    dataset: str,
    dataset_path: Optional[str],
    num_prompts: int,
    input_len: Optional[int],
    output_len: Optional[int],
    result_dir: str,
) -> List[str]:
    """Construct vllm bench serve command."""
    cmd = [
        "vllm", "bench", "serve",
        "--port", str(port),
        "--request-rate", str(request_rate),
        "--max-concurrency", str(max_concurrency),
        "--dataset", dataset,
        "--num-prompts", str(num_prompts),
        "--save-result",
        "--result-dir", result_dir,
    ]
    if dataset == "random":
        cmd.extend(["--input-len", str(input_len or 512)])
        cmd.extend(["--output-len", str(output_len or 128)])
    elif dataset_path:
        cmd.extend(["--dataset-path", dataset_path])
    return cmd


def _sample_gpu(duration: float, interval: float = 1.0) -> List[Dict[str, Any]]:
    """Sample GPU metrics at interval during benchmark. Runs in background thread."""
    samples: List[Dict[str, Any]] = []
    deadline = time.time() + duration
    while time.time() < deadline:
        gpus = utils.get_gpu_info()
        ts = utils.utc_timestamp()
        for gpu in gpus:
            samples.append({"timestamp": ts, **gpu})
        time.sleep(interval)
    return samples


def run_single(
    config: Dict[str, Any],
    request_rate: float,
    max_concurrency: int,
) -> Tuple[Path, Path]:
    """Run a single benchmark: start server, bench, collect, stop.

    Returns (run_config_path, run_metrics_path).
    """
    model = config["model"]
    port = config["port"]
    dataset = config["dataset"]
    num_prompts = config["num_prompts"]
    input_len = config.get("input_len")
    output_len = config.get("output_len")
    timeout = config["timeout"]
    results_dir = Path(config["results_dir"])
    model_safe = utils.model_safe_name(model)

    run_dir = (
        results_dir / model_safe / "sweeps" /
        f"rate_{request_rate}_conc_{max_concurrency}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    engine_version = serve.get_engine_version()
    gpu_info = utils.get_gpu_info()

    server_pid = None
    gpu_samples: List[Dict[str, Any]] = []

    try:
        # 1. START
        log_dir = run_dir
        server_pid, log_path = serve.start(
            model=model,
            port=port,
            log_dir=str(log_dir),
        )

        # 2. BENCH with GPU sampling in background
        bench_cmd = _build_bench_cmd(
            port=port,
            request_rate=request_rate,
            max_concurrency=max_concurrency,
            dataset=dataset,
            dataset_path=config.get("dataset_path"),
            num_prompts=num_prompts,
            input_len=input_len,
            output_len=output_len,
            result_dir=str(run_dir),
        )
        result_file = run_dir / "vllm_result.json"

        gpu_thread = threading.Thread(
            target=lambda: gpu_samples.extend(_sample_gpu(timeout)),
            daemon=True,
        )
        gpu_thread.start()

        subprocess.run(bench_cmd, check=True, timeout=timeout)

    finally:
        # 4. STOP -- always
        if server_pid is not None:
            serve.stop(server_pid)

    # 3. COLLECT
    run_config_data = {
        "model": model,
        "engine": config["engine"],
        "engine_version": engine_version,
        "request_rate": request_rate,
        "max_concurrency": max_concurrency,
        "dataset": dataset,
        "dataset_path": config.get("dataset_path"),
        "num_prompts": num_prompts,
        "input_len": input_len,
        "output_len": output_len,
        "port": port,
        "server_log": str(log_path),
        "gpu_info": gpu_info,
    }

    cfg_path, met_path = metrics.collect(
        result_json=result_file,
        run_config=run_config_data,
        gpu_samples=gpu_samples,
        output_dir=run_dir,
    )

    return cfg_path, met_path


def sweep(
    config: Dict[str, Any],
    request_rates: Optional[List[float]] = None,
) -> List[Tuple[Path, Path]]:
    """Run parameter sweep across all rate x concurrency combinations.

    Skips combinations that already have run_metrics.json (checkpoint/resume).
    Returns list of (run_config_path, run_metrics_path) for completed runs.
    """
    rates = request_rates or [config["request_rate"]]
    concurrencies = config["concurrencies"]
    results_dir = Path(config["results_dir"])
    model_safe = utils.model_safe_name(config["model"])

    results: List[Tuple[Path, Path]] = []
    completed = 0
    failed = 0
    total = len(rates) * len(concurrencies)

    for rate in rates:
        for conc in concurrencies:
            run_dir = (
                results_dir / model_safe / "sweeps" /
                f"rate_{rate}_conc_{conc}"
            )

            # Checkpoint: skip if already completed
            if (run_dir / "run_metrics.json").exists():
                print(f"[SKIP] rate={rate}, conc={conc} -- already complete")
                completed += 1
                continue

            print(f"[{completed + failed + 1}/{total}] rate={rate}, conc={conc}")

            try:
                cfg_path, met_path = run_single(config, rate, conc)
                results.append((cfg_path, met_path))
                completed += 1
            except Exception as e:
                print(f"[FAIL] rate={rate}, conc={conc}: {e}")
                failed += 1

    print(f"\nSweep complete: {completed} succeeded, {failed} failed, {total} total")

    # Generate plots from all completed runs
    sweep_dir = results_dir / model_safe / "sweeps"
    if results or list(sweep_dir.glob("*/run_metrics.json")):
        plot.generate(sweep_dir)

    return results

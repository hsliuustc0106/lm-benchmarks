# Refined Architecture: LM Benchmarks

**Date:** 2026-05-03
**Status:** Approved
**Context:** Review and refinement of the Universal ML Benchmarking Framework architecture before implementation. Target: LLM-only vLLM framework for a small vLLM dev team, replacing a manual `vllm bench serve` → collect → plot workflow.

## Design Principles

1. **Python-first.** Bash exists only as a thin CLI entry point. All config, metrics, orchestration, and plotting are Python.
2. **Bottom-up from real vllm output.** The framework wraps `vllm bench serve` — it doesn't reimplement benchmarking. Field names, schemas, and data shapes are driven by vllm's actual JSON output.
3. **Concrete before abstract.** No backend interface system until 2+ genuinely different backends exist. Start with LLM/vLLM code, extract abstractions later from working code.
4. **Self-describing results.** Every run writes explicit metadata (`run_config.json`). No parameters encoded in file paths or parsed from filenames.
5. **Server observability.** Server stdout/stderr are captured to disk. Health-check failures include tailed logs. GPU metrics are sampled in-run, not via a separate daemon.

## Directory Structure

```
lm-benchmarks/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── config.example.env
├── .gitignore
│
├── lm_benchmarks/
│   ├── __init__.py
│   ├── cli.py           # Click CLI: sweep, run, plot, compare, logs
│   ├── config.py        # Config loading, validation, layering
│   ├── runner.py        # Sweep orchestration
│   ├── serve.py         # vLLM server lifecycle (start, health-check, stop, log capture)
│   ├── metrics.py       # Metrics collection, statistics, schemas
│   ├── plot.py          # LLM plotting (throughput-vs-ttft, concurrency, tokens-per-user)
│   └── utils.py         # GPU info, file ops, timing
│
├── results/
│   └── .gitkeep
│
└── docs/
    ├── ARCHITECTURE.md
    ├── USAGE.md
    └── plans/
```

When omni backends arrive later, they become submodules (`lm_benchmarks/omni/`) extracted from working code.

## CLI Architecture

```
User: benchmark sweep --model <name> --rates <float...> --concurrencies <int...>
       │
       ▼
bin/benchmark          ← 5-line bash wrapper: exec python -m lm_benchmarks.cli "$@"
       │
       ▼
cli.py                 ← Click/argparse subcommands
       │
  ┌────┼────┐
  ▼    ▼    ▼
runner serve plot
  │    │
  ▼    ▼
metrics utils
```

### Subcommands

```
benchmark sweep --model <name> --rates <float...> --concurrencies <int...> \
                --dataset <name> --input-len <int> --output-len <int> \
                --num-prompts <int> --timeout <int> --results-dir <path>

benchmark run   --model <name> --rate <float> --concurrency <int> ...  # single run

benchmark plot  --results-dir <path> --output <path>

benchmark compare <dir1> <dir2>

benchmark logs  --results-dir <path>   # tail server log from last run
```

## Benchmark Pipeline

```
sweep(config)
  │
  ├─► For each (rate, concurrency) combination:
  │     │
  │     ├─► 1. START:  serve.start(model, port, log_dir)
  │     │       - vllm serve <model> --port <port> > log_dir/server.log 2>&1 &
  │     │       - Poll /health until 200 (max wait: server_timeout)
  │     │       - On timeout: tail -50 server log, include in error
  │     │       - Record: start_time, model, engine_version, server_log_path
  │     │
  │     ├─► 2. BENCH:  runner.bench(port, rate, concurrency, dataset, ...)
  │     │       - vllm bench serve --port <port> --request-rate <rate> \
  │     │           --max-concurrency <concurrency> --dataset <name> \
  │     │           --dataset-path <path> --num-prompts <N> \
  │     │           --save-result --result-dir <dir>
  │     │       - During: sample GPU metrics via nvidia-smi every 1s
  │     │       - Returns: path to vllm JSON result file
  │     │
  │     ├─► 3. COLLECT: metrics.collect(result_json, gpu_samples)
  │     │       - Read vllm output JSON
  │     │       - Extract: ttft, tpot, throughput, latencies
  │     │       - Compute: p50/p90/p95/p99 from raw latencies
  │     │       - Attach GPU metrics (avg/max util, memory)
  │     │       - Write run_config.json + run_metrics.json
  │     │
  │     ├─► 4. STOP:   serve.stop(pid)
  │     │       - SIGTERM, wait, SIGKILL if needed
  │     │       - Verify port freed
  │     │
  │     └─► 5. Checkpoint after each combination
  │            (failed sweeps resume from last completed)
  │
  └─► 6. PLOT:   plot.generate(results_dir)
        - Reads all run_metrics.json
        - Generates: throughput-vs-ttft, concurrency-scaling, tokens-per-user
        - Output: PNG in results_dir/plots/
```

### Design decisions

- **Fresh server per combination** — avoids state leakage. Optimization (`--reuse-server`) can come later.
- **GPU metrics during the run** — sampled every 1s, attached to results. No separate monitoring daemon.
- **Checkpoint per combination** — a 64-combination sweep that fails at #47 resumes from #47.

## Configuration

### Layering (later overrides earlier)

1. Built-in defaults (`config.py`)
2. `config.example.env` (project-level, committed)
3. Custom config file (`--config path/to/custom.env`)
4. CLI flags (`--rate 8`, `--concurrency 32`)

### Defaults

```python
DEFAULTS = {
    "engine": "vllm",
    "port": 8080,
    "server_timeout": 300,
    "gpu_memory_threshold": 0.9,
    "dataset": "random",
    "num_prompts": 100,
    "input_len": 8192,
    "output_len": 1024,
    "request_rate": 8.0,
    "concurrencies": [1, 2, 4, 8, 16, 32, 64],
    "timeout": 300,
    "results_dir": "./results",
    "plot_format": "png",
    "plot_dpi": 150,
}
```

### Reproducibility

Every run writes `run_config.json` containing all resolved parameters, engine version, model identifier, dataset hash, GPU topology, and timestamps. No metadata is encoded in directory names or filenames.

## Visualization

Three PNG plots per sweep, generated by `plot.py` (~150 lines, no class hierarchy):

- `throughput_vs_ttft.png` — Each rate as a series. x=TTFT (ms), y=output tokens/s.
- `concurrency_scaling.png` — Dual-axis: throughput (bars) + TTFT (line) vs concurrency.
- `tokens_per_user.png` — Tokens per concurrent user at different rates.

Reads all `run_metrics.json` files from a sweep into a pandas DataFrame, plots with matplotlib/seaborn.

## What Was Cut From the Original Spec

| Item | Reason |
|---|---|
| `backends/base/` interface system | No second backend. Extract from working code later. |
| `backends/diffusion/`, `backends/multimodal/` | Deferred until omni benchmarking needed. |
| `scripts/monitor.sh` | Replaced by in-run GPU sampling + server log capture. |
| `scripts/export_results.sh` | `benchmark plot` reads JSON directly. |
| `visualization/dashboard.py` | Overkill for team-internal use. |
| `visualization/reports/` HTML templates | Same. |
| TTS backend design | Future expansion. Don't spec now. |
| `scripts/health_check.sh` | Covered by `serve.py`. |
| `examples/` directory | README serves as example. |
| Separate docs files (INSTALLATION, API, METRICS, etc.) | Consolidate until docs outgrow 3 files. |

## Future Expansion

When omni benchmarking is needed:
- `lm_benchmarks/omni/` subpackage with omni-specific serve, benchmark, and plot modules
- Name is `omni` (not `multimodal`) to align with vLLM-Omni engine naming
- Backend interface extracted from LLM + omni working code, not designed upfront

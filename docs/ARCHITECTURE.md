# Architecture

See [Architecture Review](./plans/2026-05-03-architecture-review.md) for the full design.

## Key Design Rules

1. Python-first. Bash only in `bin/benchmark`.
2. Wraps `vllm serve` and `vllm bench serve` as subprocesses.
3. Every run writes explicit `run_config.json`. No metadata in paths.
4. Server logs captured to disk. Health-check failures include tailed logs.
5. Concrete before abstract. No backend interfaces until 2+ backends exist.

## Modules

- `cli.py` — Click CLI (sweep, run, plot, compare, logs)
- `config.py` — 4-layer config resolution
- `runner.py` — Sweep orchestration, single-run pipeline
- `serve.py` — vLLM server lifecycle (start, health-check, stop)
- `metrics.py` — vllm output parsing, statistics, result serialization
- `plot.py` — Matplotlib/seaborn plot generation
- `utils.py` — GPU info, file ops, timing

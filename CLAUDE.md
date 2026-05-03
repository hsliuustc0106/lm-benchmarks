# LM Benchmarks

Python-first benchmark framework wrapping vllm bench serve for LLM performance evaluation.

## Project layout

- `lm_benchmarks/` — Python package
- `bin/benchmark` — bash CLI entry point
- `results/` — benchmark output (git-ignored)
- `docs/plans/` — design and implementation plans

## Key design rules

1. Python-first. Bash only in `bin/benchmark` (5-line wrapper).
2. Wraps `vllm serve` and `vllm bench serve` as subprocesses. Never reimplement.
3. Every run writes explicit `run_config.json`. No metadata in paths or filenames.
4. Server logs captured to disk. Health-check failures include tailed logs.
5. Concrete before abstract. No backend interfaces until 2+ backends exist.
6. TDD: write failing test, run it, implement, run it, commit.

## Commands

- Run all tests: `python -m pytest lm_benchmarks/ -v`
- Run single test: `python -m pytest lm_benchmarks/test_<module>.py::test_name -v`
- Install in dev mode: `pip install -e ".[dev]"`

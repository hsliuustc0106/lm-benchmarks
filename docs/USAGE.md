# Usage Guide

## Running Benchmarks

### Sweep
```bash
benchmark sweep --model meta-llama/Llama-3.1-8B-Instruct \
    --rates 1 2 4 8 16 \
    --concurrencies 1 4 16 64 128 \
    --dataset sharegpt \
    --dataset-path /data/sharegpt.json \
    --num-prompts 1000
```

### Single Run
```bash
benchmark run --model meta-llama/Llama-3.1-8B-Instruct \
    --rate 8 --concurrency 32
```

## Resuming Failed Sweeps

The sweep command skips combinations that already have `run_metrics.json`. If a sweep fails partway through, re-run the same command to resume from the last completed combination.

## Results Structure

```
results/<model-safe-name>/sweeps/
├── rate_8.0_conc_1/
│   ├── run_config.json
│   ├── run_metrics.json
│   └── server.log
├── rate_8.0_conc_4/
│   └── ...
├── rate_8.0_conc_16/
│   └── ...
└── plots/
    ├── throughput_vs_ttft.png
    ├── concurrency_scaling_rate_8.0.png
    └── tokens_per_user.png
```

## Datasets

- `random` — synthetic random prompts (uses `--input-len` and `--output-len`)
- `sharegpt` — ShareGPT conversations (requires `--dataset-path`)
- `sonnet` — Sonnet dataset (requires `--dataset-path`)
- Custom path — any JSON dataset file

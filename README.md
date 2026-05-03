# LM Benchmarks

vLLM benchmark framework. Wraps `vllm bench serve` with parameter sweeps, metrics collection, and plotting.

## Quick Start

```bash
pip install -e ".[dev]"

# Single benchmark
benchmark run --model meta-llama/Llama-3.1-8B-Instruct --rate 8 --concurrency 32

# Parameter sweep
benchmark sweep --model meta-llama/Llama-3.1-8B-Instruct \
    --rates 1 2 4 8 --concurrencies 1 4 16 32 64

# Generate plots
benchmark plot results/meta-llama__Llama-3.1-8B-Instruct/sweeps/

# View server logs from last run
benchmark logs results/
```

## Configuration

Config is resolved in 4 layers (later overrides earlier):
1. Built-in defaults
2. `.env` file (`--config-file`)
3. Environment variables (uppercase: `REQUEST_RATE=16`)
4. CLI flags (`--rate 16`)

See `config.example.env` for available options.

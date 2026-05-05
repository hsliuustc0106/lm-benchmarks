"""CLI entry point for lm-benchmarks."""
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import click

from lm_benchmarks import config, runner
from lm_benchmarks import serve as serve_mod
from lm_benchmarks import plot as plot_mod
from lm_benchmarks.utils import model_safe_name


@click.group()
def main():
    """LM Benchmarks — vLLM benchmark framework."""
    pass


@main.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--model", required=True, help="Model identifier")
@click.option("--port", type=int, default=None, help="Server port")
@click.pass_context
def serve(ctx: click.Context, model: str, port: Optional[int]):
    """Start vllm server with managed logging.

    All unknown arguments are forwarded directly to vllm serve.

    Examples:

        benchmark serve --model Qwen/Qwen3.6-35B-A3B

        benchmark serve --model Qwen/Qwen3.6-35B-A3B --max-model-len 131072

        benchmark serve --model Qwen/Qwen3.6-35B-A3B --tensor-parallel-size 4 --port 8081
    """
    cli_overrides = {"model": model}
    if port is not None:
        cli_overrides["port"] = port

    cfg = config.load(cli_overrides=cli_overrides)
    port = cfg["port"]
    results_path = Path(cfg["results_dir"])
    model_safe = model_safe_name(model)
    log_dir = results_path / model_safe
    log_dir.mkdir(parents=True, exist_ok=True)

    extra_args = list(ctx.args) if ctx.args else None
    pid, log_path = serve_mod.start(
        model=model,
        port=port,
        log_dir=str(log_dir),
        additional_args=extra_args,
        timeout=cfg["server_timeout"],
    )

    click.echo(f"vLLM server started (PID {pid})")
    click.echo(f"Log: {log_path}")
    click.echo("--- server log ---")

    try:
        subprocess.call(["tail", "-f", str(log_path)])
    except FileNotFoundError:
        input("Press Enter to stop server...")
    finally:
        serve_mod.stop(pid)
        click.echo("Server stopped.")


@main.command()
@click.option("--model", required=True, help="Model identifier")
@click.option("--rates", "-r", multiple=True, type=float, help="Request rates")
@click.option("--concurrencies", "-c", multiple=True, type=int, help="Concurrency levels")
@click.option("--dataset", default=None, help="Dataset name (random, sharegpt, sonnet, or path)")
@click.option("--dataset-path", default=None, help="Path to dataset file")
@click.option("--input-len", type=int, default=None, help="Input token length (random dataset)")
@click.option("--output-len", type=int, default=None, help="Output token length (random dataset)")
@click.option("--num-prompts", type=int, default=None, help="Number of prompts")
@click.option("--timeout", type=int, default=None, help="Benchmark timeout in seconds")
@click.option("--config-file", default=None, help="Path to .env config file")
@click.option("--no-server", is_flag=True, default=False, help="Skip starting vllm server (use existing)")
@click.option("--server-log", default=None, help="Path to external vllm server log (used with --no-server)")
def sweep(
    model: str,
    rates: tuple,
    concurrencies: tuple,
    dataset: Optional[str],
    dataset_path: Optional[str],
    input_len: Optional[int],
    output_len: Optional[int],
    num_prompts: Optional[int],
    timeout: Optional[int],
    config_file: Optional[str],
    no_server: bool,
    server_log: Optional[str],
):
    """Run parameter sweep across rate x concurrency combinations."""
    cli_overrides = {"model": model}
    if dataset:
        cli_overrides["dataset"] = dataset
    if dataset_path:
        cli_overrides["dataset_path"] = dataset_path
    if input_len is not None:
        cli_overrides["input_len"] = input_len
    if output_len is not None:
        cli_overrides["output_len"] = output_len
    if num_prompts is not None:
        cli_overrides["num_prompts"] = num_prompts
    if timeout is not None:
        cli_overrides["timeout"] = timeout

    cfg = config.load(env_file=config_file, cli_overrides=cli_overrides)

    request_rates = list(rates) if rates else None
    if concurrencies:
        cfg["concurrencies"] = list(concurrencies)

    runner.sweep(cfg, request_rates=request_rates, no_server=no_server, server_log_path=server_log)


@main.command()
@click.option("--model", required=True, help="Model identifier")
@click.option("--rate", "-r", type=float, required=True, help="Request rate")
@click.option("--concurrency", "-c", type=int, required=True, help="Max concurrency")
@click.option("--dataset", default=None)
@click.option("--dataset-path", default=None)
@click.option("--input-len", type=int, default=None)
@click.option("--output-len", type=int, default=None)
@click.option("--num-prompts", type=int, default=None)
@click.option("--timeout", type=int, default=None, help="Benchmark timeout in seconds")
@click.option("--config-file", default=None)
@click.option("--no-server", is_flag=True, default=False, help="Skip starting vllm server (use existing)")
@click.option("--server-log", default=None, help="Path to external vllm server log (used with --no-server)")
def run(
    model: str,
    rate: float,
    concurrency: int,
    dataset: Optional[str],
    dataset_path: Optional[str],
    input_len: Optional[int],
    output_len: Optional[int],
    num_prompts: Optional[int],
    timeout: Optional[int],
    config_file: Optional[str],
    no_server: bool,
    server_log: Optional[str],
):
    """Run a single benchmark."""
    cli_overrides = {"model": model}
    if dataset:
        cli_overrides["dataset"] = dataset
    if dataset_path:
        cli_overrides["dataset_path"] = dataset_path
    if input_len is not None:
        cli_overrides["input_len"] = input_len
    if output_len is not None:
        cli_overrides["output_len"] = output_len
    if num_prompts is not None:
        cli_overrides["num_prompts"] = num_prompts
    if timeout is not None:
        cli_overrides["timeout"] = timeout

    cfg = config.load(env_file=config_file, cli_overrides=cli_overrides)

    cfg_path, met_path = runner.run_single(cfg, rate, concurrency, no_server=no_server, server_log_path=server_log)
    click.echo(f"Config: {cfg_path}")
    click.echo(f"Metrics: {met_path}")


@main.command()
@click.argument("results_dir", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output directory for plots")
@click.option("--heatmap", is_flag=True, default=False, help="Generate heatmaps for TTFT, TPOT, throughput")
@click.option("--tput-vs-tpu", "tput_tpu", is_flag=True, default=False, help="Generate throughput vs tokens-per-user plot")
def plot(results_dir: str, output: Optional[str], heatmap: bool, tput_tpu: bool):
    """Generate plots from sweep results."""
    sweep_dir = Path(results_dir)
    output_dir = Path(output) if output else sweep_dir / "plots"

    if not heatmap and not tput_tpu:
        # Default: existing standard plots
        plot_mod.generate(sweep_dir)
        return

    df = plot_mod._load_sweep_results(sweep_dir)
    if df is None or df.empty:
        click.echo("No metrics found")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if tput_tpu:
        path = output_dir / "throughput_vs_tokens_per_user.png"
        plot_mod.plot_throughput_vs_tpu(df, path)
        click.echo(f"  {path}")

    if heatmap:
        for metric in ["mean_ttft_ms", "mean_tpot_ms", "output_throughput"]:
            path = output_dir / f"heatmap_{metric}.png"
            plot_mod.plot_heatmap(df, path, metric=metric)
            click.echo(f"  {path}")


@main.command()
@click.argument("left", type=click.Path(exists=True))
@click.argument("right", type=click.Path(exists=True))
@click.option("--output", "-o", default="results/comparison.png", help="Output PNG path")
@click.option("--label-left", default="Left", help="Label for left sweep")
@click.option("--label-right", default="Right", help="Label for right sweep")
def compare(left: str, right: str, output: str, label_left: str, label_right: str):
    """Overlay two sweep result directories on shared axes."""
    plot_mod.compare_sweeps(
        [Path(left), Path(right)],
        labels=[label_left, label_right],
        output=Path(output),
    )
    click.echo(f"Comparison saved to {output}")


@main.command()
@click.argument("results_dir", type=click.Path(exists=True))
def logs(results_dir: str):
    """Show server logs from the most recent run."""
    import os
    log_files = sorted(
        Path(results_dir).rglob("server.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not log_files:
        click.echo("No server logs found.")
        return

    latest = log_files[0]
    click.echo(f"Server log: {latest}\n")
    with open(latest) as f:
        click.echo(f.read())


if __name__ == "__main__":
    main()

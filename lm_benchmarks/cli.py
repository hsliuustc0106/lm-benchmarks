"""CLI entry point for lm-benchmarks."""
import sys
from pathlib import Path
from typing import List, Optional

import click

from lm_benchmarks import config, runner, serve
from lm_benchmarks import plot as plot_mod
from lm_benchmarks.utils import model_safe_name


@click.group()
def main():
    """LM Benchmarks — vLLM benchmark framework."""
    pass


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
@click.option("--results-dir", default=None, help="Results directory")
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
    results_dir: Optional[str],
):
    """Run parameter sweep across rate x concurrency combinations."""
    cli_overrides = {"model": model}
    if results_dir:
        cli_overrides["results_dir"] = results_dir
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

    runner.sweep(cfg, request_rates=request_rates)


@main.command()
@click.option("--model", required=True, help="Model identifier")
@click.option("--rate", "-r", type=float, required=True, help="Request rate")
@click.option("--concurrency", "-c", type=int, required=True, help="Max concurrency")
@click.option("--dataset", default=None)
@click.option("--dataset-path", default=None)
@click.option("--input-len", type=int, default=None)
@click.option("--output-len", type=int, default=None)
@click.option("--num-prompts", type=int, default=None)
@click.option("--config-file", default=None)
@click.option("--results-dir", default=None)
def run(
    model: str,
    rate: float,
    concurrency: int,
    dataset: Optional[str],
    dataset_path: Optional[str],
    input_len: Optional[int],
    output_len: Optional[int],
    num_prompts: Optional[int],
    config_file: Optional[str],
    results_dir: Optional[str],
):
    """Run a single benchmark."""
    cli_overrides = {"model": model}
    if results_dir:
        cli_overrides["results_dir"] = results_dir
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

    cfg = config.load(env_file=config_file, cli_overrides=cli_overrides)

    cfg_path, met_path = runner.run_single(cfg, rate, concurrency)
    click.echo(f"Config: {cfg_path}")
    click.echo(f"Metrics: {met_path}")


@main.command()
@click.argument("results_dir", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output directory for plots")
def plot(results_dir: str, output: Optional[str]):
    """Generate plots from sweep results."""
    sweep_dir = Path(results_dir)
    plot_mod.generate(sweep_dir)


@main.command()
@click.argument("left", type=click.Path(exists=True))
@click.argument("right", type=click.Path(exists=True))
def compare(left: str, right: str):
    """Compare two sweep result directories."""
    import json

    left_dir = Path(left)
    right_dir = Path(right)

    left_metrics = list(left_dir.glob("**/run_metrics.json"))
    right_metrics = list(right_dir.glob("**/run_metrics.json"))

    click.echo(f"Left:  {len(left_metrics)} runs in {left}")
    click.echo(f"Right: {len(right_metrics)} runs in {right}")

    click.echo("Detailed comparison coming soon.")


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

"""Plot generation: throughput-vs-ttft, concurrency scaling, tokens per user."""
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lm_benchmarks.utils import load_json


def _load_sweep_results(sweep_dir: Path) -> Optional[pd.DataFrame]:
    """Load all run_metrics.json files from a sweep directory into a DataFrame."""
    rows = []
    for metrics_file in sweep_dir.glob("*/run_metrics.json"):
        data = load_json(metrics_file)
        if data is None:
            continue
        rows.append({
            "request_rate": data.get("request_rate"),
            "max_concurrency": data.get("max_concurrency"),
            "mean_ttft_ms": data.get("mean_ttft_ms"),
            "output_throughput": data.get("output_throughput"),
            "mean_tpot_ms": data.get("mean_tpot_ms"),
        })

    if not rows:
        return None

    return pd.DataFrame(rows)


def _plot_throughput_vs_ttft(df: pd.DataFrame, output: Path) -> None:
    """Scatter plot: output throughput vs mean TTFT, colored by request rate."""
    fig, ax = plt.subplots(figsize=(12, 8))

    for rate in sorted(df["request_rate"].unique()):
        subset = df[df["request_rate"] == rate]
        ax.scatter(
            subset["mean_ttft_ms"], subset["output_throughput"],
            s=100, alpha=0.7, label=f"Rate {rate}",
        )

    ax.set_xlabel("Mean TTFT (ms)")
    ax.set_ylabel("Output Throughput (tokens/s)")
    ax.set_title("Output Throughput vs Time to First Token")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_concurrency_scaling(df: pd.DataFrame, output: Path) -> None:
    """Dual-axis plot: throughput (bars) + TTFT (line) vs concurrency."""
    subset = df.sort_values("max_concurrency")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color1 = "steelblue"
    ax1.bar(
        subset["max_concurrency"].astype(str),
        subset["output_throughput"],
        color=color1, alpha=0.7,
    )
    ax1.set_xlabel("Max Concurrency")
    ax1.set_ylabel("Output Throughput (tokens/s)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "coral"
    ax2.plot(
        subset["max_concurrency"].astype(str),
        subset["mean_ttft_ms"],
        "o-", color=color2, linewidth=2, markersize=8,
    )
    ax2.set_ylabel("Mean TTFT (ms)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("Concurrency Scaling")
    ax1.grid(True, alpha=0.3)

    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_tokens_per_user(df: pd.DataFrame, output: Path) -> None:
    """Bar chart: tokens per user across concurrency levels."""
    df = df.copy()
    df["tokens_per_user"] = df["output_throughput"] / df["max_concurrency"]

    fig, ax = plt.subplots(figsize=(12, 6))

    rates = sorted(df["request_rate"].unique())
    x = sorted(df["max_concurrency"].unique())
    width = 0.8 / len(rates)

    for i, rate in enumerate(rates):
        subset = df[df["request_rate"] == rate].set_index("max_concurrency")
        values = [subset.loc[c, "tokens_per_user"] if c in subset.index else 0 for c in x]
        offset = (i - len(rates) / 2 + 0.5) * width
        ax.bar([str(v) for v in x], values, width, label=f"Rate {rate}")

    ax.set_xlabel("Max Concurrency")
    ax.set_ylabel("Tokens per User")
    ax.set_title("Tokens per Concurrent User")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate(sweep_dir: Path) -> None:
    """Generate all plots for a sweep directory."""
    df = _load_sweep_results(sweep_dir)
    if df is None or df.empty:
        print(f"No metrics found in {sweep_dir}")
        return

    plot_dir = sweep_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")

    _plot_throughput_vs_ttft(df, plot_dir / "throughput_vs_ttft.png")
    _plot_tokens_per_user(df, plot_dir / "tokens_per_user.png")

    # Concurrency scaling generates one plot per request rate
    for rate in sorted(df["request_rate"].unique()):
        subset = df[df["request_rate"] == rate]
        _plot_concurrency_scaling(subset, plot_dir / f"concurrency_scaling_rate_{rate}.png")

    print(f"Plots saved to {plot_dir}")

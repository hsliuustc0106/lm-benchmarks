import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import lm_benchmarks.plot as plot


def make_sweep_dir(tmp_path):
    """Create a sweep directory with run_metrics.json files."""
    sweep = tmp_path / "sweeps"
    sweep.mkdir(parents=True)

    runs = [
        ("rate_8.0_conc_1", 8.0, 1, 200.0, 4000.0),
        ("rate_8.0_conc_4", 8.0, 4, 350.0, 3800.0),
        ("rate_8.0_conc_16", 8.0, 16, 600.0, 3500.0),
        ("rate_4.0_conc_1", 4.0, 1, 150.0, 4200.0),
        ("rate_4.0_conc_4", 4.0, 4, 250.0, 4000.0),
        ("rate_4.0_conc_16", 4.0, 16, 450.0, 3700.0),
    ]

    for name, rate, conc, ttft, throughput in runs:
        d = sweep / name
        d.mkdir()
        (d / "run_metrics.json").write_text(json.dumps({
            "request_rate": rate,
            "max_concurrency": conc,
            "mean_ttft_ms": ttft,
            "output_throughput": throughput,
            "mean_tpot_ms": 25.0,
        }))

    return sweep


def test_load_sweep_results_builds_dataframe(tmp_path):
    """_load_sweep_results returns a DataFrame with all metrics."""
    sweep = make_sweep_dir(tmp_path)
    df = plot._load_sweep_results(sweep)
    assert len(df) == 6
    assert list(df.columns) == [
        "request_rate", "max_concurrency", "mean_ttft_ms",
        "output_throughput", "mean_tpot_ms",
    ]


def test_generate_creates_plot_files(tmp_path):
    """generate() creates PNG files in plots/ subdirectory."""
    sweep = make_sweep_dir(tmp_path)
    plot_dir = sweep / "plots"

    with patch("matplotlib.pyplot.savefig"):
        with patch("matplotlib.pyplot.figure"):
            with patch("matplotlib.pyplot.subplots") as mock_subplots:
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_subplots.return_value = (mock_fig, mock_ax)
                plot.generate(sweep)

    # Directory was created
    assert plot_dir.exists()


def test_generate_skips_when_no_metrics(tmp_path):
    """generate() returns early if no run_metrics.json found."""
    sweep = tmp_path / "empty_sweep"
    sweep.mkdir()
    plot.generate(sweep)  # should not crash


# --- compare_sweeps ---

def test_compare_sweeps_creates_plot_files(tmp_path):
    """compare_sweeps creates a comparison PNG with multiple lines."""
    d1 = make_sweep_dir(tmp_path / "sweep_a")
    d2 = make_sweep_dir(tmp_path / "sweep_b")
    out = tmp_path / "comparison.png"

    plot.compare_sweeps([d1, d2], labels=["8K Input", "1K Input"], output=out)

    assert out.exists()
    assert out.stat().st_size > 0


def test_compare_sweeps_requires_matching_labels(tmp_path):
    """compare_sweeps raises when sweep_dirs and labels length differ."""
    d = make_sweep_dir(tmp_path / "sweep")
    try:
        plot.compare_sweeps([d], labels=["A", "B"], output=tmp_path / "x.png")
        assert False, "should have raised"
    except ValueError:
        pass


# --- throughput vs tokens-per-user ---

def test_plot_throughput_vs_tpu_creates_plot(tmp_path):
    """plot_throughput_vs_tpu generates a scatter PNG."""
    df = _make_sweep_df(make_sweep_dir(tmp_path / "sweep"))
    out = tmp_path / "tput_vs_tpu.png"

    plot.plot_throughput_vs_tpu(df, out)

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_throughput_vs_tpu_empty( tmp_path):
    """plot_throughput_vs_tpu with empty DataFrame does not crash."""
    import pandas as pd
    df = pd.DataFrame()
    out = tmp_path / "empty.png"

    plot.plot_throughput_vs_tpu(df, out)

    assert not out.exists()  # no data → no plot


# --- heatmap ---

def test_plot_heatmap_creates_plot(tmp_path):
    """plot_heatmap generates an annotated heatmap PNG."""
    df = _make_sweep_df(make_sweep_dir(tmp_path / "sweep"))
    out = tmp_path / "heatmap.png"

    plot.plot_heatmap(df, out, metric="mean_ttft_ms")

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_heatmap_empty(tmp_path):
    """plot_heatmap with empty DataFrame does not crash."""
    import pandas as pd
    df = pd.DataFrame()
    out = tmp_path / "empty_heat.png"

    plot.plot_heatmap(df, out, metric="mean_ttft_ms")

    assert not out.exists()


# --- helpers for tests ---

def _make_sweep_df(sweep_dir):
    """Build a DataFrame like _load_sweep_results would produce."""
    return plot._load_sweep_results(sweep_dir)

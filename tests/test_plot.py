import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import lm_benchmarks.plot as plot


def make_sweep_dir(tmp_path):
    """Create a sweep directory with run_metrics.json files."""
    sweep = tmp_path / "sweeps"
    sweep.mkdir()

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

"""End-to-end integration test with mocked subprocesses."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from lm_benchmarks.cli import main


SAMPLE_VLLM_RESULT = {
    "mean_ttft_ms": 450.0,
    "median_ttft_ms": 380.0,
    "p99_ttft_ms": 1200.0,
    "mean_tpot_ms": 25.0,
    "median_tpot_ms": 24.0,
    "p99_tpot_ms": 55.0,
    "request_throughput": 7.8,
    "output_throughput": 4200.0,
    "total_token_throughput": 38000.0,
}


def test_sweep_end_to_end(tmp_path):
    """Full sweep pipeline: config -> serve -> bench -> collect -> plot."""
    results_dir = tmp_path / "results"

    with patch("lm_benchmarks.runner.serve") as mock_serve:
        mock_serve.start.return_value = (12345, Path("/tmp/server.log"))
        mock_serve.get_engine_version.return_value = "vllm 0.11.0"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            # Write fake vllm result to the right path after subprocess.run
            def write_fake_result(*args, **kwargs):
                cmd = kwargs.get("args", args[0] if args else [])
                for i, arg in enumerate(cmd):
                    if arg == "--result-dir" and i + 1 < len(cmd):
                        run_dir = Path(cmd[i + 1])
                        run_dir.mkdir(parents=True, exist_ok=True)
                        with open(run_dir / "vllm_result.json", "w") as f:
                            json.dump(SAMPLE_VLLM_RESULT, f)
                return MagicMock(returncode=0)

            mock_run.side_effect = write_fake_result

            with patch("lm_benchmarks.runner.utils.get_gpu_info", return_value=[]):
                with patch("lm_benchmarks.plot.plt.savefig"):
                    with patch("lm_benchmarks.plot.plt.figure"):
                        with patch("lm_benchmarks.plot.plt.subplots") as mock_sub:
                            mock_sub.return_value = (MagicMock(), MagicMock())

                            runner = CliRunner()
                            result = runner.invoke(main, [
                                "sweep",
                                "--model", "test/model",
                                "--rates", "8.0",
                                "--concurrencies", "1",
                                "--concurrencies", "4",
                                "--num-prompts", "10",
                                "--results-dir", str(results_dir),
                            ])

    assert result.exit_code == 0

    # Verify results were written
    run_dirs = list(results_dir.rglob("run_metrics.json"))
    assert len(run_dirs) == 2  # (8.0, 1) and (8.0, 4)

    for metrics_file in run_dirs:
        with open(metrics_file) as f:
            m = json.load(f)
            assert m["mean_ttft_ms"] == 450.0
            assert m["output_throughput"] == 4200.0

    # Verify run_config.json exists for each run
    config_files = list(results_dir.rglob("run_config.json"))
    assert len(config_files) == 2
    for cf in config_files:
        with open(cf) as f:
            c = json.load(f)
            assert c["model"] == "test/model"
            assert "run_id" in c
            assert "started_at" in c


def test_run_single_end_to_end(tmp_path):
    """Single benchmark via CLI."""
    results_dir = tmp_path / "results"

    with patch("lm_benchmarks.runner.serve") as mock_serve:
        mock_serve.start.return_value = (12345, Path("/tmp/server.log"))
        mock_serve.get_engine_version.return_value = "vllm 0.11.0"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            # Pre-create the result file that vllm would write
            def write_fake_result(*args, **kwargs):
                cmd = kwargs.get("args", args[0] if args else [])
                for i, arg in enumerate(cmd):
                    if arg == "--result-dir" and i + 1 < len(cmd):
                        run_dir = Path(cmd[i + 1])
                        run_dir.mkdir(parents=True, exist_ok=True)
                        with open(run_dir / "vllm_result.json", "w") as f:
                            json.dump(SAMPLE_VLLM_RESULT, f)
                return MagicMock(returncode=0)

            mock_run.side_effect = write_fake_result

            with patch("lm_benchmarks.runner.utils.get_gpu_info", return_value=[]):
                runner = CliRunner()
                result = runner.invoke(main, [
                    "run",
                    "--model", "test/model",
                    "--rate", "8.0",
                    "--concurrency", "32",
                    "--num-prompts", "10",
                    "--results-dir", str(results_dir),
                ])

    assert result.exit_code == 0
    assert "Metrics:" in result.output

import json
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import lm_benchmarks.runner as runner


def make_config(**overrides):
    cfg = {
        "model": "test/model",
        "port": 9999,
        "dataset": "random",
        "num_prompts": 10,
        "input_len": 512,
        "output_len": 128,
        "request_rate": 8.0,
        "concurrencies": [1, 4],
        "timeout": 60,
        "results_dir": "/tmp/results",
        "engine": "vllm",
    }
    cfg.update(overrides)
    return cfg


def test_build_bench_command():
    """_build_bench_cmd constructs correct vllm bench serve command."""
    cmd = runner._build_bench_cmd(
        port=8080,
        request_rate=8.0,
        max_concurrency=32,
        dataset="sharegpt",
        dataset_path="/data/sharegpt.json",
        num_prompts=100,
        input_len=None,
        output_len=None,
        result_dir="/tmp/results",
    )
    assert "vllm" in cmd[0]
    assert "bench" in cmd
    assert "serve" in cmd
    assert "--port" in cmd
    assert "8080" in cmd
    assert "--request-rate" in cmd
    assert "8.0" in cmd
    assert "--max-concurrency" in cmd
    assert "32" in cmd
    assert "--dataset" in cmd
    assert "sharegpt" in cmd
    assert "--dataset-path" in cmd
    assert "/data/sharegpt.json" in cmd
    assert "--num-prompts" in cmd
    assert "100" in cmd
    assert "--save-result" in cmd
    assert "--result-dir" in cmd


def test_build_bench_command_random_dataset_skips_path():
    """Random dataset doesn't need --dataset-path."""
    cmd = runner._build_bench_cmd(
        port=8080, request_rate=8.0, max_concurrency=4,
        dataset="random", dataset_path=None,
        num_prompts=10, input_len=512, output_len=128,
        result_dir="/tmp",
    )
    assert "--dataset-path" not in cmd


def test_build_bench_command_random_includes_input_output_len():
    """Random dataset includes --input-len and --output-len."""
    cmd = runner._build_bench_cmd(
        port=8080, request_rate=8.0, max_concurrency=4,
        dataset="random", dataset_path=None,
        num_prompts=10, input_len=512, output_len=128,
        result_dir="/tmp",
    )
    assert "--input-len" in cmd
    assert "512" in cmd
    assert "--output-len" in cmd
    assert "128" in cmd


def test_run_single_benchmark(tmp_path):
    """run_single executes the full start -> bench -> collect -> stop pipeline."""
    with patch("lm_benchmarks.runner.serve") as mock_serve:
        mock_serve.start.return_value = (12345, Path("/tmp/server.log"))
        mock_serve.get_engine_version.return_value = "vllm 0.11.0"

        with patch("lm_benchmarks.runner.metrics") as mock_metrics:
            mock_metrics.collect.return_value = (
                Path("/tmp/run_config.json"),
                Path("/tmp/run_metrics.json"),
            )

            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_run.return_value = mock_result

                with patch("lm_benchmarks.runner.utils.get_gpu_info", return_value=[]):
                    cfg_path, metrics_path = runner.run_single(
                        config=make_config(results_dir=str(tmp_path)),
                        request_rate=8.0,
                        max_concurrency=32,
                    )

    assert cfg_path is not None
    assert metrics_path is not None
    mock_serve.start.assert_called_once()
    mock_serve.stop.assert_called_once_with(12345)
    mock_run.assert_called_once()


def test_run_single_stops_server_even_on_bench_failure(tmp_path):
    """Server is always stopped, even if benchmark fails."""
    with patch("lm_benchmarks.runner.serve") as mock_serve:
        mock_serve.start.return_value = (12345, Path("/tmp/server.log"))
        mock_serve.get_engine_version.return_value = "vllm 0.11.0"

        with patch("subprocess.run", side_effect=Exception("bench crashed")):
            with patch("lm_benchmarks.runner.utils.get_gpu_info", return_value=[]):
                import pytest
                with pytest.raises(Exception, match="bench crashed"):
                    runner.run_single(
                        config=make_config(results_dir=str(tmp_path)),
                        request_rate=8.0,
                        max_concurrency=32,
                    )

    mock_serve.stop.assert_called_once_with(12345)


def test_sweep_iterates_over_combinations(tmp_path):
    """sweep() runs all rate x concurrency combinations."""
    call_args = []

    def fake_run_single(config, request_rate, max_concurrency):
        call_args.append((request_rate, max_concurrency))
        return Path(f"/tmp/run_{request_rate}_{max_concurrency}.json"), Path("/tmp/metrics.json")

    with patch("lm_benchmarks.runner.run_single", side_effect=fake_run_single):
        with patch("lm_benchmarks.runner.plot") as mock_plot:
            runner.sweep(
                config=make_config(
                    request_rate=8.0,
                    concurrencies=[1, 4],
                    results_dir=str(tmp_path),
                ),
                request_rates=[8.0],
            )

    assert len(call_args) == 2
    assert (8.0, 1) in call_args
    assert (8.0, 4) in call_args
    mock_plot.generate.assert_called_once()


def test_sweep_resumes_from_checkpoint(tmp_path):
    """sweep() skips combinations that already have run_metrics.json."""
    # Pre-create metrics for (8.0, 1) to simulate a completed run
    # Must match the path the runner constructs: results_dir / model_safe / "sweeps" / "rate_{rate}_conc_{conc}"
    done_dir = tmp_path / "test__model" / "sweeps" / "rate_8.0_conc_1"
    done_dir.mkdir(parents=True)
    (done_dir / "run_metrics.json").write_text("{}")

    call_args = []

    def fake_run_single(config, request_rate, max_concurrency):
        call_args.append((request_rate, max_concurrency))
        out_dir = tmp_path / f"rate_{request_rate}_conc_{max_concurrency}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return Path("/tmp/cfg.json"), Path("/tmp/met.json")

    with patch("lm_benchmarks.runner.run_single", side_effect=fake_run_single):
        with patch("lm_benchmarks.runner.plot"):
            runner.sweep(
                config=make_config(
                    request_rate=8.0,
                    concurrencies=[1, 4],
                    results_dir=str(tmp_path),
                ),
                request_rates=[8.0],
            )

    # Only (8.0, 4) should run; (8.0, 1) was already done
    assert call_args == [(8.0, 4)]


def test_gpu_sampling_happens_during_benchmark(tmp_path):
    """GPU metrics are sampled while benchmark runs."""
    gpu_samples = []

    def fake_get_gpu_info():
        gpu_samples.append(1)
        return [{"utilization_pct": 80}]

    with patch("lm_benchmarks.runner.serve") as mock_serve:
        mock_serve.start.return_value = (12345, Path("/tmp/server.log"))
        mock_serve.get_engine_version.return_value = "vllm 0.11.0"

        with patch("lm_benchmarks.runner.metrics") as mock_metrics:
            mock_metrics.collect.return_value = (Path("/tmp/cfg.json"), Path("/tmp/met.json"))

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                with patch("lm_benchmarks.runner.utils.get_gpu_info",
                          side_effect=fake_get_gpu_info):
                    with patch("lm_benchmarks.runner.time.sleep"):
                        runner.run_single(
                            config=make_config(results_dir=str(tmp_path)),
                            request_rate=8.0,
                            max_concurrency=4,
                        )

    # GPU is sampled (background thread runs during subprocess)
    assert len(gpu_samples) > 0

import subprocess
import time
from unittest.mock import patch, MagicMock, call
import lm_benchmarks.serve as serve
import requests


def test_start_constructs_correct_command():
    """start() launches vllm serve with correct arguments."""
    with patch("subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        with patch.object(serve, "_wait_for_health", return_value=True):
            pid, log_path = serve.start(
                model="meta-llama/Llama-3.1-8B",
                port=8080,
                log_dir="/tmp/logs",
                additional_args=["--tensor-parallel-size", "4"],
            )

    mock_popen.assert_called_once()
    cmd = mock_popen.call_args[0][0]
    assert "vllm" in cmd[0] or cmd[0].endswith("vllm")
    assert "serve" in cmd
    assert "meta-llama/Llama-3.1-8B" in cmd
    assert "--port" in cmd
    assert "8080" in cmd
    assert "--tensor-parallel-size" in cmd
    assert "4" in cmd
    assert pid == 12345


def test_start_redirects_stdout_stderr_to_log():
    """Server stdout/stderr go to the log file."""
    with patch("subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        with patch.object(serve, "_wait_for_health", return_value=True):
            serve.start("model", log_dir="/tmp/logs")

    kwargs = mock_popen.call_args[1]
    assert "stdout" in kwargs or "stderr" in kwargs


def test_start_health_check_polls_until_ready():
    """_wait_for_health polls /health until 200."""
    responses = [
        requests.ConnectionError("Connection refused"),
        MagicMock(status_code=503),  # loading
        MagicMock(status_code=200),  # ready
    ]
    with patch("requests.get", side_effect=responses):
        with patch("time.sleep"):  # don't actually sleep
            result = serve._wait_for_health(port=8080, timeout=300)
    assert result is True


def test_start_health_check_timeout_returns_false():
    """_wait_for_health returns False after timeout."""
    with patch("requests.get", side_effect=requests.ConnectionError("Connection refused")):  # never ready
        with patch("time.sleep"):
            result = serve._wait_for_health(port=8080, timeout=1)
    assert result is False


def test_start_raises_on_health_check_failure():
    """start() raises RuntimeError if server never becomes healthy."""
    with patch("subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        with patch.object(serve, "_wait_for_health", return_value=False):
            import pytest
            with pytest.raises(RuntimeError, match="health check"):
                serve.start("model", port=8080, log_dir="/tmp/logs")


def test_stop_sends_sigterm_then_sigkill():
    """stop() tries SIGTERM first, then SIGKILL."""
    with patch("os.kill") as mock_kill:
        with patch("psutil.Process") as mock_psutil:
            mock_proc = MagicMock()
            mock_proc.is_running.side_effect = [True, True, False]
            mock_psutil.return_value = mock_proc

            with patch("time.sleep"):
                serve.stop(12345)

    assert call(12345, 15) in mock_kill.call_args_list, "SIGTERM not sent"


def test_stop_handles_already_dead_process():
    """stop() doesn't crash if process is already gone."""
    with patch("os.kill", side_effect=ProcessLookupError):
        serve.stop(12345)  # should not raise


def test_get_engine_version():
    """get_engine_version() returns vllm version string."""
    with patch("subprocess.check_output", return_value="vllm 0.11.0\n"):
        version = serve.get_engine_version()
        assert "0.11.0" in version


def test_get_engine_version_handles_missing():
    with patch("subprocess.check_output", side_effect=FileNotFoundError):
        version = serve.get_engine_version()
        assert version == "unknown"

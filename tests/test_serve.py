import subprocess
import time
import signal
from unittest.mock import patch, MagicMock, call
import lm_benchmarks.serve as serve
import requests


def test_start_constructs_correct_command():
    """start() launches vllm serve with correct arguments."""
    with patch("shutil.which", return_value="/usr/bin/vllm"):
        with patch("lm_benchmarks.serve.cleanup_gpu_processes"):
            with patch("lm_benchmarks.serve.cleanup_port"):
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
        # First Popen call is vllm serve
        cmd = mock_popen.call_args[0][0]
        assert "vllm" in cmd[0]
    assert "serve" in cmd
    assert "meta-llama/Llama-3.1-8B" in cmd
    assert "--port" in cmd
    assert "8080" in cmd
    assert "--tensor-parallel-size" in cmd
    assert "4" in cmd
    assert pid == 12345


def test_start_redirects_stdout_stderr_to_log():
    """Server stdout/stderr go to the log file."""
    with patch("lm_benchmarks.serve.cleanup_gpu_processes"):
        with patch("lm_benchmarks.serve.cleanup_port"):
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
    with patch("lm_benchmarks.serve.cleanup_gpu_processes"):
        with patch("lm_benchmarks.serve.cleanup_port"):
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


def test_find_vllm_processes():
    """_find_vllm_processes returns PIDs of vllm serve processes."""
    with patch("psutil.process_iter") as mock_iter:
        mock_proc_1 = MagicMock()
        mock_proc_1.info = {"pid": 100, "name": "python", "cmdline": ["vllm", "serve", "--model", "test"]}
        mock_proc_2 = MagicMock()
        mock_proc_2.info = {"pid": 200, "name": "python", "cmdline": ["other", "process"]}
        mock_iter.return_value = [mock_proc_1, mock_proc_2]
        pids = serve._find_vllm_processes()
    assert pids == [100]


def test_find_vllm_processes_empty():
    """_find_vllm_processes returns empty list when no vllm processes."""
    with patch("psutil.process_iter") as mock_iter:
        mock_iter.return_value = []
        pids = serve._find_vllm_processes()
    assert pids == []


def test_cleanup_gpu_processes_kills_vllm():
    """cleanup_gpu_processes terminates all vllm serve processes."""
    with patch("lm_benchmarks.serve._find_vllm_processes") as mock_find:
        mock_find.side_effect = [[100, 200], []]  # first: found, second: none survive
        with patch("os.kill") as mock_kill:
            serve.cleanup_gpu_processes()
    mock_kill.assert_any_call(100, signal.SIGTERM)
    mock_kill.assert_any_call(200, signal.SIGTERM)


def test_cleanup_gpu_processes_sigkill_survivors():
    """cleanup_gpu_processes sends SIGKILL to survivors after SIGTERM."""
    with patch("lm_benchmarks.serve._find_vllm_processes") as mock_find:
        mock_find.side_effect = [[100], [100]]  # still alive after SIGTERM
        with patch("os.kill") as mock_kill:
            serve.cleanup_gpu_processes()
    mock_kill.assert_any_call(100, signal.SIGTERM)
    mock_kill.assert_any_call(100, signal.SIGKILL)


def test_cleanup_port_noop_when_port_free():
    """cleanup_port does nothing when no process is on the port."""
    with patch("psutil.net_connections", return_value=[]):
        with patch("subprocess.run") as mock_run:
            serve.cleanup_port(9999)
    mock_run.assert_not_called()


def test_cleanup_port_kills_process_on_port():
    """cleanup_port terminates a process found on the given port."""
    mock_conn = MagicMock()
    mock_conn.laddr.port = 8080
    mock_conn.pid = 54321
    mock_proc = MagicMock()
    with patch("psutil.net_connections", return_value=[mock_conn]):
        with patch("psutil.Process", return_value=mock_proc):
            serve.cleanup_port(8080)
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once_with(timeout=5)


def test_start_cleans_up_port():
    """start() calls cleanup_gpu_processes and cleanup_port before launching vllm."""
    with patch("lm_benchmarks.serve.cleanup_gpu_processes") as mock_gpu_cleanup:
        with patch("lm_benchmarks.serve.cleanup_port") as mock_port_cleanup:
            with patch("subprocess.Popen") as mock_popen:
                mock_process = MagicMock()
                mock_process.pid = 12345
                mock_popen.return_value = mock_process
                with patch.object(serve, "_wait_for_health", return_value=True):
                    serve.start("test/model", port=8080, log_dir="/tmp/logs")

    mock_gpu_cleanup.assert_called_once()
    mock_port_cleanup.assert_called_once_with(8080)


def test_get_engine_version():
    """get_engine_version() returns vllm version string."""
    with patch("subprocess.check_output", return_value="vllm 0.11.0\n"):
        version = serve.get_engine_version()
        assert "0.11.0" in version


def test_get_engine_version_handles_missing():
    with patch("subprocess.check_output", side_effect=FileNotFoundError):
        version = serve.get_engine_version()
        assert version == "unknown"

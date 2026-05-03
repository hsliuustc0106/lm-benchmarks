"""vLLM server lifecycle: start, health-check, stop, version query."""
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests


def _find_vllm_processes() -> List[int]:
    """Return PIDs of any running vllm serve processes."""
    pids = []
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                if any("vllm" in part and "serve" in cmdline for part in cmdline):
                    pids.append(proc.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        try:
            result = subprocess.run(
                ["pgrep", "-f", "vllm.*serve"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                pids = [int(pid) for pid in result.stdout.strip().split()]
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
    return pids


def cleanup_gpu_processes() -> None:
    """Kill any existing vllm serve processes to free GPUs."""
    pids = _find_vllm_processes()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
    if pids:
        time.sleep(3)
        # SIGKILL survivors
        for pid in _find_vllm_processes():
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass


def cleanup_port(port: int) -> None:
    """Kill any process listening on the given port."""
    try:
        import psutil
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr.port == port and conn.pid:
                proc = psutil.Process(conn.pid)
                proc.terminate()
                proc.wait(timeout=5)
    except (ImportError, psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
        # Fallback: try fuser
        try:
            subprocess.run(
                ["fuser", "-k", f"{port}/tcp"],
                capture_output=True, timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass


def start(
    model: str,
    port: int = 8080,
    log_dir: Optional[str] = None,
    additional_args: Optional[List[str]] = None,
) -> Tuple[int, Path]:
    """Start vLLM serving. Returns (pid, log_path). Raises RuntimeError if unhealthy."""
    cleanup_gpu_processes()
    cleanup_port(port)

    log_dir = Path(log_dir) if log_dir else Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "server.log"

    cmd = ["vllm", "serve", model, "--port", str(port)]
    if additional_args:
        cmd.extend(additional_args)

    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    healthy = _wait_for_health(port, timeout=300)

    if not healthy:
        # Capture tail of log for diagnostics
        log_file.close()
        with open(log_path) as f:
            lines = f.readlines()
            tail = "".join(lines[-50:]) if lines else "(empty log)"
        _kill_process(proc.pid)
        raise RuntimeError(
            f"vLLM server failed health check on port {port}.\n"
            f"Last 50 lines of server log:\n{tail}"
        )

    return proc.pid, log_path


def _wait_for_health(port: int, timeout: int) -> bool:
    """Poll /health until 200 or timeout expires."""
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    return False


def stop(pid: int) -> None:
    """Stop the server process. SIGTERM first, then SIGKILL."""
    _kill_process(pid)
    time.sleep(2)
    try:
        os.kill(pid, 0)
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


def _kill_process(pid: int) -> None:
    """Send SIGTERM to process. Ignores if already dead."""
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass


def get_engine_version() -> str:
    """Return vllm version string."""
    try:
        result = subprocess.check_output(["vllm", "--version"], text=True, timeout=10)
        return result.strip()
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        return "unknown"

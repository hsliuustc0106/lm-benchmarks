# LM Benchmarks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an LLM-only vLLM benchmark framework that automates the `vllm bench serve` → collect metrics → plot pipeline with parameter sweeps.

**Architecture:** Python package (`lm_benchmarks`) with a thin bash CLI entry point. Seven modules: cli, config, runner, serve, metrics, plot, utils. Wraps `vllm serve` and `vllm bench serve` as subprocesses, captures server logs and GPU metrics in-run, writes self-describing JSON results.

**Tech Stack:** Python 3.10+, click, pandas, matplotlib, seaborn, pytest

---

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `config.example.env`
- Create: `lm_benchmarks/__init__.py`
- Create: `results/.gitkeep`
- Create: `CLAUDE.md`
- Create: `bin/benchmark`

**Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "lm-benchmarks"
version = "0.1.0"
description = "LLM benchmark framework wrapping vllm bench serve"
requires-python = ">=3.10"
dependencies = [
    "click>=8.0",
    "pandas>=2.0",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "requests>=2.31",
]

[project.scripts]
benchmark = "lm_benchmarks.cli:main"

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-mock>=3.10",
]
```

**Step 2: Write requirements.txt**

```
click>=8.0
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
requests>=2.31
```

**Step 3: Write .gitignore**

```
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
results/*
!results/.gitkeep
.env
*.swp
.DS_Store
```

**Step 4: Write config.example.env**

```bash
# LM Benchmarks configuration
# Copy to .env and customize

# Serving
# ENGINE=vllm
# PORT=8080
# SERVER_TIMEOUT=300
# GPU_MEMORY_THRESHOLD=0.9

# Benchmark defaults
# DATASET=random
# NUM_PROMPTS=100
# INPUT_LEN=8192
# OUTPUT_LEN=1024
# REQUEST_RATE=8.0
# TIMEOUT=300

# Results
# RESULTS_DIR=./results
# PLOT_FORMAT=png
# PLOT_DPI=150
```

**Step 5: Write lm_benchmarks/__init__.py**

```python
"""LM Benchmarks — vLLM benchmark framework."""
__version__ = "0.1.0"
```

**Step 6: Write results/.gitkeep**

```bash
touch results/.gitkeep
```

**Step 7: Write CLAUDE.md**

```markdown
# LM Benchmarks

Python-first benchmark framework wrapping vllm bench serve for LLM performance evaluation.

## Project layout

- `lm_benchmarks/` — Python package
- `bin/benchmark` — bash CLI entry point
- `results/` — benchmark output (git-ignored)
- `docs/plans/` — design and implementation plans

## Key design rules

1. Python-first. Bash only in `bin/benchmark` (5-line wrapper).
2. Wraps `vllm serve` and `vllm bench serve` as subprocesses. Never reimplement.
3. Every run writes explicit `run_config.json`. No metadata in paths or filenames.
4. Server logs captured to disk. Health-check failures include tailed logs.
5. Concrete before abstract. No backend interfaces until 2+ backends exist.
6. TDD: write failing test, run it, implement, run it, commit.

## Commands

- Run all tests: `python -m pytest lm_benchmarks/ -v`
- Run single test: `python -m pytest lm_benchmarks/test_<module>.py::test_name -v`
- Install in dev mode: `pip install -e ".[dev]"`
```

**Step 8: Write bin/benchmark**

```bash
#!/usr/bin/env bash
# Entry point for lm-benchmarks. Delegates everything to Python.
exec python -m lm_benchmarks.cli "$@"
```

```bash
chmod +x bin/benchmark
```

**Step 9: Verify scaffold**

```bash
python -c "import lm_benchmarks; print(lm_benchmarks.__version__)"
```
Expected: `0.1.0`

**Step 10: Commit**

```bash
git add -A
git commit -m "chore: scaffold project structure"
```

---

### Task 2: Config Module

**Files:**
- Create: `lm_benchmarks/config.py`
- Create: `lm_benchmarks/test_config.py`

**Step 1: Write the failing test**

```python
# lm_benchmarks/test_config.py
import os
import tempfile
from pathlib import Path
import lm_benchmarks.config as config


def test_defaults_are_set():
    """Built-in defaults exist for all required keys."""
    cfg = config.load()
    assert cfg["engine"] == "vllm"
    assert cfg["port"] == 8080
    assert cfg["server_timeout"] == 300
    assert cfg["dataset"] == "random"
    assert cfg["num_prompts"] == 100
    assert cfg["input_len"] == 8192
    assert cfg["output_len"] == 1024
    assert cfg["request_rate"] == 8.0
    assert cfg["concurrencies"] == [1, 2, 4, 8, 16, 32, 64]
    assert cfg["timeout"] == 300
    assert cfg["results_dir"] == "./results"


def test_env_var_overrides_default():
    """Environment variables override defaults."""
    os.environ["REQUEST_RATE"] = "42.0"
    cfg = config.load()
    assert cfg["request_rate"] == 42.0
    del os.environ["REQUEST_RATE"]


def test_dotenv_file_overrides_defaults():
    """A .env file overrides built-in defaults."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("NUM_PROMPTS=500\nINPUT_LEN=4096\n")
        env_path = f.name
    try:
        cfg = config.load(env_path)
        assert cfg["num_prompts"] == 500
        assert cfg["input_len"] == 4096
    finally:
        os.unlink(env_path)


def test_cli_overrides_override_all():
    """CLI overrides take highest precedence."""
    os.environ["REQUEST_RATE"] = "10.0"
    cli_overrides = {"request_rate": 99.0, "num_prompts": 1}
    cfg = config.load(cli_overrides=cli_overrides)
    assert cfg["request_rate"] == 99.0
    assert cfg["num_prompts"] == 1
    del os.environ["REQUEST_RATE"]


def test_unknown_key_ignored():
    """Keys not in DEFAULTS are ignored (no crash)."""
    os.environ["BOGUS_KEY"] = "should_be_ignored"
    cfg = config.load()
    assert "BOGUS_KEY" not in cfg
    del os.environ["BOGUS_KEY"]


def test_int_coercion():
    """String values are coerced to the type of the default."""
    os.environ["NUM_PROMPTS"] = "999"
    cfg = config.load()
    assert cfg["num_proMPTS"] == 999
    assert isinstance(cfg["num_proMPTS"], int)
    del os.environ["NUM_PROMPTS"]


def test_float_coercion():
    os.environ["REQUEST_RATE"] = "3.5"
    cfg = config.load()
    assert cfg["request_rate"] == 3.5
    assert isinstance(cfg["request_rate"], float)
    del os.environ["REQUEST_RATE"]


def test_concurrencies_parsing():
    """Concurrencies from env are parsed as list of ints."""
    os.environ["CONCURRENCIES"] = "1 4 16 64"
    cfg = config.load()
    assert cfg["concurrencies"] == [1, 4, 16, 64]
    del os.environ["CONCURRENCIES"]
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest lm_benchmarks/test_config.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'lm_benchmarks.config'`

**Step 3: Write minimal implementation**

```python
# lm_benchmarks/config.py
"""Configuration loading with 4-layer precedence: defaults < env file < env vars < CLI."""
import os
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULTS: Dict[str, Any] = {
    "engine": "vllm",
    "port": 8080,
    "server_timeout": 300,
    "gpu_memory_threshold": 0.9,
    "dataset": "random",
    "num_prompts": 100,
    "input_len": 8192,
    "output_len": 1024,
    "request_rate": 8.0,
    "concurrencies": [1, 2, 4, 8, 16, 32, 64],
    "timeout": 300,
    "results_dir": "./results",
    "plot_format": "png",
    "plot_dpi": 150,
}

LIST_KEYS = {"concurrencies"}


def _coerce(value: str, default: Any) -> Any:
    """Coerce a string value to the type of the default."""
    if isinstance(default, bool):
        return value.lower() in ("true", "1", "yes")
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, list):
        return [type(default[0])(v) if default else v for v in value.split()]
    return value


def load(
    env_file: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load config with layered precedence.

    Precedence (lowest to highest):
    1. Built-in DEFAULTS
    2. Env file (dotenv-style, optional)
    3. Environment variables
    4. CLI overrides dict
    """
    cfg = dict(DEFAULTS)

    # Layer 2: env file
    if env_file and Path(env_file).exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip().lower()
                value = value.strip().strip("\"'")
                if key in DEFAULTS:
                    cfg[key] = _coerce(value, DEFAULTS[key])

    # Layer 3: environment variables
    for key in DEFAULTS:
        env_val = os.environ.get(key.upper())
        if env_val is not None:
            cfg[key] = _coerce(env_val, DEFAULTS[key])

    # Layer 4: CLI overrides
    if cli_overrides:
        for key, value in cli_overrides.items():
            if key in DEFAULTS:
                cfg[key] = value

    return cfg
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest lm_benchmarks/test_config.py -v
```
Expected: PASS (8 tests)

**Step 5: Commit**

```bash
git add lm_benchmarks/config.py lm_benchmarks/test_config.py
git commit -m "feat: add config module with 4-layer precedence"
```

---

### Task 3: Utils Module

**Files:**
- Create: `lm_benchmarks/utils.py`
- Create: `lm_benchmarks/test_utils.py`

**Step 1: Write the failing test**

```python
# lm_benchmarks/test_utils.py
import time
import json
from unittest.mock import patch, MagicMock
import lm_benchmarks.utils as utils


def test_timestamp_format():
    ts = utils.utc_timestamp()
    assert "T" in ts
    assert ts.endswith("Z") or "+" in ts


def test_run_id_is_unique():
    id1 = utils.generate_run_id()
    time.sleep(0.01)
    id2 = utils.generate_run_id()
    assert id1 != id2
    assert len(id1) > 10


def test_gpu_info_parses_nvidia_smi_output():
    sample_output = """0, 85, 45000, 81559, 65
1, 72, 42000, 81559, 62
"""
    with patch("subprocess.check_output", return_value=sample_output):
        gpus = utils.get_gpu_info()
        assert len(gpus) == 2
        assert gpus[0]["index"] == 0
        assert gpus[0]["utilization_pct"] == 85
        assert gpus[0]["memory_used_mb"] == 45000
        assert gpus[0]["memory_total_mb"] == 81559
        assert gpus[0]["temperature_c"] == 65


def test_gpu_info_handles_nvidia_smi_missing():
    with patch("subprocess.check_output", side_effect=FileNotFoundError):
        gpus = utils.get_gpu_info()
        assert gpus == []


def test_gpu_info_handles_subprocess_error():
    with patch("subprocess.check_output", side_effect=Exception("no GPU")):
        gpus = utils.get_gpu_info()
        assert gpus == []


def test_save_json_writes_file(tmp_path):
    data = {"key": "value", "num": 42}
    path = tmp_path / "test.json"
    utils.save_json(path, data)
    assert path.exists()
    with open(path) as f:
        assert json.load(f) == data


def test_load_json_reads_file(tmp_path):
    data = {"a": 1, "b": [2, 3]}
    path = tmp_path / "test.json"
    with open(path, "w") as f:
        json.dump(data, f)
    result = utils.load_json(path)
    assert result == data


def test_load_json_returns_none_for_missing_file(tmp_path):
    result = utils.load_json(tmp_path / "nonexistent.json")
    assert result is None


def test_model_safe_name():
    assert utils.model_safe_name("meta-llama/Llama-3.1-8B") == "meta-llama__Llama-3.1-8B"
    assert utils.model_safe_name("simple-model") == "simple-model"
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest lm_benchmarks/test_utils.py -v
```
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# lm_benchmarks/utils.py
"""Utility functions: GPU info, file ops, timing, run identity."""
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_timestamp() -> str:
    """ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def generate_run_id() -> str:
    """Unique run identifier: YYYYMMDDTHHMMSS-<uuid8>."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"{ts}-{short}"


def get_gpu_info() -> List[Dict[str, Any]]:
    """Query nvidia-smi for GPU topology. Returns empty list on failure."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired, Exception):
        return []

    gpus = []
    for line in output.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        gpus.append({
            "index": int(parts[0]),
            "utilization_pct": int(parts[1]),
            "memory_used_mb": int(parts[2]),
            "memory_total_mb": int(parts[3]),
            "temperature_c": int(parts[4]),
        })
    return gpus


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """Write dict to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read JSON file. Returns None if missing or unparseable."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def model_safe_name(model: str) -> str:
    """Convert model identifier to filesystem-safe name."""
    return model.replace("/", "__").replace(":", "__")
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest lm_benchmarks/test_utils.py -v
```
Expected: PASS (9 tests)

**Step 5: Commit**

```bash
git add lm_benchmarks/utils.py lm_benchmarks/test_utils.py
git commit -m "feat: add utils module"
```

---

### Task 4: Serve Module

**Files:**
- Create: `lm_benchmarks/serve.py`
- Create: `lm_benchmarks/test_serve.py`

**Step 1: Write the failing test**

```python
# lm_benchmarks/test_serve.py
import subprocess
import time
from unittest.mock import patch, MagicMock, call
import lm_benchmarks.serve as serve


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
        None,  # connection refused
        MagicMock(status_code=503),  # loading
        MagicMock(status_code=200),  # ready
    ]
    with patch("requests.get", side_effect=responses):
        with patch("time.sleep"):  # don't actually sleep
            result = serve._wait_for_health(port=8080, timeout=300)
    assert result is True


def test_start_health_check_timeout_returns_false():
    """_wait_for_health returns False after timeout."""
    with patch("requests.get", return_value=None):  # never ready
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

    assert any(call(12345, 15) in mock_kill.call_args_list
               for call in [mock_kill.call_args_list]), "SIGTERM not sent"


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
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest lm_benchmarks/test_serve.py -v
```
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# lm_benchmarks/serve.py
"""vLLM server lifecycle: start, health-check, stop, version query."""
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests


def start(
    model: str,
    port: int = 8080,
    log_dir: Optional[str] = None,
    additional_args: Optional[List[str]] = None,
) -> Tuple[int, Path]:
    """Start vLLM serving. Returns (pid, log_path). Raises RuntimeError if unhealthy."""
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
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest lm_benchmarks/test_serve.py -v
```
Expected: PASS (9 tests)

**Step 5: Commit**

```bash
git add lm_benchmarks/serve.py lm_benchmarks/test_serve.py
git commit -m "feat: add serve module for vLLM server lifecycle"
```

---

### Task 5: Metrics Module

**Files:**
- Create: `lm_benchmarks/metrics.py`
- Create: `lm_benchmarks/test_metrics.py`

**Step 1: Write the failing test**

```python
# lm_benchmarks/test_metrics.py
import json
from unittest.mock import patch
import lm_benchmarks.metrics as metrics


SAMPLE_VLLM_OUTPUT = {
    "mean_ttft_ms": 450.2,
    "median_ttft_ms": 380.5,
    "p99_ttft_ms": 1200.0,
    "mean_tpot_ms": 25.3,
    "median_tpot_ms": 24.1,
    "p99_tpot_ms": 55.0,
    "request_throughput": 7.8,
    "output_throughput": 4200.0,
    "total_token_throughput": 38000.0,
}


def test_collect_reads_vllm_json(tmp_path):
    """collect() reads vllm output JSON and extracts metrics."""
    result_file = tmp_path / "result.json"
    with open(result_file, "w") as f:
        json.dump(SAMPLE_VLLM_OUTPUT, f)

    run_config = {"model": "test/model", "request_rate": 8.0, "max_concurrency": 32}
    gpu_samples = [
        {"timestamp": 0, "utilization_pct": 85, "memory_used_mb": 40000},
        {"timestamp": 1, "utilization_pct": 90, "memory_used_mb": 41000},
    ]

    run_cfg_path, metrics_path = metrics.collect(
        result_json=result_file,
        run_config=run_config,
        gpu_samples=gpu_samples,
        output_dir=tmp_path,
    )

    assert run_cfg_path.exists()
    assert metrics_path.exists()

    with open(metrics_path) as f:
        m = json.load(f)
        assert m["mean_ttft_ms"] == 450.2
        assert m["mean_tpot_ms"] == 25.3
        assert m["output_throughput"] == 4200.0
        assert m["gpu_avg_utilization_pct"] == 87.5


def test_collect_computes_tokens_per_user():
    """metrics includes tokens_per_user = output_throughput / concurrency."""
    result_file = tmp_path / "result.json"
    with open(result_file, "w") as f:
        json.dump({"output_throughput": 4000.0, "request_throughput": 8.0}, f)

    run_config = {"max_concurrency": 16}
    _, metrics_path = metrics.collect(
        result_file, run_config, [], output_dir=tmp_path
    )

    with open(metrics_path) as f:
        m = json.load(f)
    assert m["tokens_per_user"] == 250.0  # 4000 / 16


def test_collect_handles_missing_vllm_fields():
    """Missing fields default to None, don't crash."""
    result_file = tmp_path / "result.json"
    with open(result_file, "w") as f:
        json.dump({}, f)

    _, metrics_path = metrics.collect(
        result_file, {}, [], output_dir=tmp_path
    )

    with open(metrics_path) as f:
        m = json.load(f)
    assert m["mean_ttft_ms"] is None


def test_run_config_includes_timestamps():
    """Written run_config includes started_at and completed_at."""
    result_file = tmp_path / "result.json"
    with open(result_file, "w") as f:
        json.dump({}, f)

    cfg_path, _ = metrics.collect(
        result_file, {"model": "x"}, [], output_dir=tmp_path
    )

    with open(cfg_path) as f:
        cfg = json.load(f)
    assert "run_id" in cfg
    assert "started_at" in cfg
    assert "completed_at" in cfg
    assert cfg["model"] == "x"


def test_gpu_metrics_empty_when_no_samples():
    """GPU stats are None when no samples provided."""
    result_file = tmp_path / "result.json"
    with open(result_file, "w") as f:
        json.dump(SAMPLE_VLLM_OUTPUT, f)

    _, metrics_path = metrics.collect(
        result_file, {}, [], output_dir=tmp_path
    )

    with open(metrics_path) as f:
        m = json.load(f)
    assert m["gpu_avg_utilization_pct"] is None


def test_compute_percentiles():
    """_compute_percentiles returns p50, p90, p95, p99."""
    data = list(range(1, 101))  # 1..100
    result = metrics._compute_percentiles(data)
    assert result["p50"] == 50.5  # median of 1..100
    assert result["p95"] == 95.05
    assert "p90" in result
    assert "p99" in result


def test_summary_stats():
    """_summary_stats returns mean, min, max, count."""
    result = metrics._summary_stats([10, 20, 30])
    assert result["mean"] == 20.0
    assert result["min"] == 10.0
    assert result["max"] == 30.0
    assert result["count"] == 3
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest lm_benchmarks/test_metrics.py -v
```
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# lm_benchmarks/metrics.py
"""Metrics collection: read vllm output, compute statistics, write results."""
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lm_benchmarks.utils import generate_run_id, load_json, save_json, utc_timestamp


def collect(
    result_json: Path,
    run_config: Dict[str, Any],
    gpu_samples: List[Dict[str, Any]],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Process vllm benchmark output into run_config.json and run_metrics.json.

    Returns (run_config_path, run_metrics_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_json(result_json) or {}

    # Build run_config with resolved parameters and metadata
    cfg = dict(run_config)
    cfg["run_id"] = generate_run_id()
    cfg["started_at"] = utc_timestamp()
    cfg["completed_at"] = utc_timestamp()
    cfg["result_file"] = str(result_json)

    cfg_path = output_dir / "run_config.json"
    save_json(cfg_path, cfg)

    # Build metrics
    concurrency = run_config.get("max_concurrency", 1) or 1
    output_throughput = raw.get("output_throughput")

    gpu_utils = [s["utilization_pct"] for s in gpu_samples if "utilization_pct" in s]

    metrics = {
        "run_id": cfg["run_id"],
        "model": run_config.get("model"),
        "request_rate": run_config.get("request_rate"),
        "max_concurrency": run_config.get("max_concurrency"),
        "mean_ttft_ms": raw.get("mean_ttft_ms"),
        "median_ttft_ms": raw.get("median_ttft_ms"),
        "p99_ttft_ms": raw.get("p99_ttft_ms"),
        "mean_tpot_ms": raw.get("mean_tpot_ms"),
        "median_tpot_ms": raw.get("median_tpot_ms"),
        "p99_tpot_ms": raw.get("p99_tpot_ms"),
        "request_throughput": raw.get("request_throughput"),
        "output_throughput": output_throughput,
        "total_token_throughput": raw.get("total_token_throughput"),
        "tokens_per_user": output_throughput / concurrency if output_throughput else None,
        "gpu_avg_utilization_pct": statistics.mean(gpu_utils) if gpu_utils else None,
        "gpu_max_utilization_pct": max(gpu_utils) if gpu_utils else None,
    }

    metrics_path = output_dir / "run_metrics.json"
    save_json(metrics_path, metrics)

    return cfg_path, metrics_path


def _compute_percentiles(values: List[float]) -> Dict[str, float]:
    """Compute p50, p90, p95, p99 from a list of values."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def pct(p):
        k = (n - 1) * p / 100
        f = int(k)
        c = k - f
        if f + 1 < n:
            return sorted_vals[f] + c * (sorted_vals[f + 1] - sorted_vals[f])
        return sorted_vals[f]

    return {
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
    }


def _summary_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean, min, max, count."""
    if not values:
        return {"mean": 0, "min": 0, "max": 0, "count": 0}
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest lm_benchmarks/test_metrics.py -v
```
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add lm_benchmarks/metrics.py lm_benchmarks/test_metrics.py
git commit -m "feat: add metrics module for vllm output processing"
```

---

### Task 6: Runner Module

**Files:**
- Create: `lm_benchmarks/runner.py`
- Create: `lm_benchmarks/test_runner.py`

**Step 1: Write the failing test**

```python
# lm_benchmarks/test_runner.py
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
    """sweep() runs all rate × concurrency combinations."""
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
    done_dir = tmp_path / "rate_8.0_conc_1"
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
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest lm_benchmarks/test_runner.py -v
```
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# lm_benchmarks/runner.py
"""Benchmark orchestration: sweep, single run, GPU sampling."""
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lm_benchmarks import metrics, serve, utils
from lm_benchmarks import plot as plot_mod


def _build_bench_cmd(
    port: int,
    request_rate: float,
    max_concurrency: int,
    dataset: str,
    dataset_path: Optional[str],
    num_prompts: int,
    input_len: Optional[int],
    output_len: Optional[int],
    result_dir: str,
) -> List[str]:
    """Construct vllm bench serve command."""
    cmd = [
        "vllm", "bench", "serve",
        "--port", str(port),
        "--request-rate", str(request_rate),
        "--max-concurrency", str(max_concurrency),
        "--dataset", dataset,
        "--num-prompts", str(num_prompts),
        "--save-result",
        "--result-dir", result_dir,
    ]
    if dataset == "random":
        cmd.extend(["--input-len", str(input_len or 512)])
        cmd.extend(["--output-len", str(output_len or 128)])
    elif dataset_path:
        cmd.extend(["--dataset-path", dataset_path])
    return cmd


def _sample_gpu(duration: float, interval: float = 1.0) -> List[Dict[str, Any]]:
    """Sample GPU metrics at interval during benchmark. Runs in background thread."""
    samples: List[Dict[str, Any]] = []
    deadline = time.time() + duration
    while time.time() < deadline:
        gpus = utils.get_gpu_info()
        ts = utils.utc_timestamp()
        for gpu in gpus:
            samples.append({"timestamp": ts, **gpu})
        time.sleep(interval)
    return samples


def run_single(
    config: Dict[str, Any],
    request_rate: float,
    max_concurrency: int,
) -> Tuple[Path, Path]:
    """Run a single benchmark: start server, bench, collect, stop.

    Returns (run_config_path, run_metrics_path).
    """
    model = config["model"]
    port = config["port"]
    dataset = config["dataset"]
    num_prompts = config["num_prompts"]
    input_len = config.get("input_len")
    output_len = config.get("output_len")
    timeout = config["timeout"]
    results_dir = Path(config["results_dir"])
    model_safe = utils.model_safe_name(model)

    run_dir = (
        results_dir / model_safe / "sweeps" /
        f"rate_{request_rate}_conc_{max_concurrency}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    engine_version = serve.get_engine_version()
    gpu_info = utils.get_gpu_info()

    server_pid = None
    gpu_samples: List[Dict[str, Any]] = []

    try:
        # 1. START
        log_dir = run_dir
        server_pid, log_path = serve.start(
            model=model,
            port=port,
            log_dir=str(log_dir),
        )

        # 2. BENCH with GPU sampling in background
        bench_cmd = _build_bench_cmd(
            port=port,
            request_rate=request_rate,
            max_concurrency=max_concurrency,
            dataset=dataset,
            dataset_path=config.get("dataset_path"),
            num_prompts=num_prompts,
            input_len=input_len,
            output_len=output_len,
            result_dir=str(run_dir),
        )
        result_file = run_dir / "vllm_result.json"

        gpu_thread = threading.Thread(
            target=lambda: gpu_samples.extend(_sample_gpu(timeout)),
            daemon=True,
        )
        gpu_thread.start()

        subprocess.run(bench_cmd, check=True, timeout=timeout)

    finally:
        # 4. STOP — always
        if server_pid is not None:
            serve.stop(server_pid)

    # 3. COLLECT
    run_config_data = {
        "model": model,
        "engine": config["engine"],
        "engine_version": engine_version,
        "request_rate": request_rate,
        "max_concurrency": max_concurrency,
        "dataset": dataset,
        "dataset_path": config.get("dataset_path"),
        "num_prompts": num_prompts,
        "input_len": input_len,
        "output_len": output_len,
        "port": port,
        "server_log": str(log_path),
        "gpu_info": gpu_info,
    }

    cfg_path, met_path = metrics.collect(
        result_json=result_file,
        run_config=run_config_data,
        gpu_samples=gpu_samples,
        output_dir=run_dir,
    )

    return cfg_path, met_path


def sweep(
    config: Dict[str, Any],
    request_rates: Optional[List[float]] = None,
) -> List[Tuple[Path, Path]]:
    """Run parameter sweep across all rate × concurrency combinations.

    Skips combinations that already have run_metrics.json (checkpoint/resume).
    Returns list of (run_config_path, run_metrics_path) for completed runs.
    """
    rates = request_rates or [config["request_rate"]]
    concurrencies = config["concurrencies"]
    results_dir = Path(config["results_dir"])
    model_safe = utils.model_safe_name(config["model"])

    results: List[Tuple[Path, Path]] = []
    completed = 0
    failed = 0
    total = len(rates) * len(concurrencies)

    for rate in rates:
        for conc in concurrencies:
            run_dir = (
                results_dir / model_safe / "sweeps" /
                f"rate_{rate}_conc_{conc}"
            )

            # Checkpoint: skip if already completed
            if (run_dir / "run_metrics.json").exists():
                print(f"[SKIP] rate={rate}, conc={conc} — already complete")
                completed += 1
                continue

            print(f"[{completed + failed + 1}/{total}] rate={rate}, conc={conc}")

            try:
                cfg_path, met_path = run_single(config, rate, conc)
                results.append((cfg_path, met_path))
                completed += 1
            except Exception as e:
                print(f"[FAIL] rate={rate}, conc={conc}: {e}")
                failed += 1

    print(f"\nSweep complete: {completed} succeeded, {failed} failed, {total} total")

    # Generate plots from all completed runs
    sweep_dir = results_dir / model_safe / "sweeps"
    if results or list(sweep_dir.glob("*/run_metrics.json")):
        plot_mod.generate(sweep_dir)

    return results
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest lm_benchmarks/test_runner.py -v
```
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add lm_benchmarks/runner.py lm_benchmarks/test_runner.py
git commit -m "feat: add runner module for benchmark orchestration"
```

---

### Task 7: Plot Module

**Files:**
- Create: `lm_benchmarks/plot.py`
- Create: `lm_benchmarks/test_plot.py`

**Step 1: Write the failing test**

```python
# lm_benchmarks/test_plot.py
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
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest lm_benchmarks/test_plot.py -v
```
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# lm_benchmarks/plot.py
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
    rates = sorted(df["request_rate"].unique())

    for rate in rates:
        subset = df[df["request_rate"] == rate].sort_values("max_concurrency")

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

        ax1.set_title(f"Concurrency Scaling (Rate {rate} req/s)")
        ax1.grid(True, alpha=0.3)

        out = output.parent / f"concurrency_scaling_rate_{rate}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
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
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest lm_benchmarks/test_plot.py -v
```
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add lm_benchmarks/plot.py lm_benchmarks/test_plot.py
git commit -m "feat: add plot module for benchmark visualization"
```

---

### Task 8: CLI Module

**Files:**
- Create: `lm_benchmarks/cli.py`

**Step 1: Write the CLI implementation**

Since the CLI is a thin wiring layer over already-tested modules, write it directly (no separate test file needed — the integration test in Task 9 covers the CLI).

```python
# lm_benchmarks/cli.py
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
    """Run parameter sweep across rate × concurrency combinations."""
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

    # TODO: more detailed comparison (side-by-side plots, diff table)
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
```

**Step 2: Verify CLI loads**

```bash
python -m lm_benchmarks.cli --help
```
Expected: Help text with sweep, run, plot, compare, logs subcommands.

**Step 3: Commit**

```bash
git add lm_benchmarks/cli.py
git commit -m "feat: add CLI module with sweep, run, plot, compare, logs"
```

---

### Task 9: Integration Test & Final Wiring

**Files:**
- Create: `lm_benchmarks/test_integration.py`

**Step 1: Write integration test**

```python
# lm_benchmarks/test_integration.py
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
    """Full sweep pipeline: config → serve → bench → collect → plot."""
    results_dir = tmp_path / "results"

    with patch("lm_benchmarks.runner.serve") as mock_serve:
        mock_serve.start.return_value = (12345, Path("/tmp/server.log"))
        mock_serve.get_engine_version.return_value = "vllm 0.11.0"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            # Write fake vllm result to the right path after subprocess.run
            def write_fake_result(*args, **kwargs):
                # vllm bench serve writes result to --result-dir
                cmd = kwargs.get("args", args[0] if args else [])
                for i, arg in enumerate(cmd):
                    if arg == "--result-dir" and i + 1 < len(cmd):
                        result_dir = Path(cmd[i + 1])
                        result_dir.mkdir(parents=True, exist_ok=True)
                        with open(result_dir / "result.json", "w") as f:
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
                                "--concurrencies", "1", "4",
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
                        result_dir = Path(cmd[i + 1])
                        result_dir.mkdir(parents=True, exist_ok=True)
                        with open(result_dir / "result.json", "w") as f:
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
```

**Step 2: Run integration tests**

```bash
python -m pytest lm_benchmarks/test_integration.py -v
```
Expected: PASS (2 tests)

**Step 3: Commit**

```bash
git add lm_benchmarks/test_integration.py
git commit -m "test: add integration tests for end-to-end pipeline"
```

---

### Task 10: Documentation

**Files:**
- Create: `README.md`
- Create: `docs/ARCHITECTURE.md`
- Create: `docs/USAGE.md`

**Step 1: Write README.md**

```markdown
# LM Benchmarks

vLLM benchmark framework. Wraps `vllm bench serve` with parameter sweeps, metrics collection, and plotting.

## Quick Start

```bash
pip install -e ".[dev]"

# Single benchmark
benchmark run --model meta-llama/Llama-3.1-8B-Instruct --rate 8 --concurrency 32

# Parameter sweep
benchmark sweep --model meta-llama/Llama-3.1-8B-Instruct \
    --rates 1 2 4 8 --concurrencies 1 4 16 32 64

# Generate plots
benchmark plot results/meta-llama__Llama-3.1-8B-Instruct/sweeps/

# View server logs from last run
benchmark logs results/
```

## Configuration

Config is resolved in 4 layers (later overrides earlier):
1. Built-in defaults
2. `.env` file (`--config-file`)
3. Environment variables (uppercase: `REQUEST_RATE=16`)
4. CLI flags (`--rate 16`)

See `config.example.env` for available options.
```

**Step 2: Write docs/ARCHITECTURE.md**

Reference the approved architecture review document.

```markdown
# Architecture

See [Architecture Review](./plans/2026-05-03-architecture-review.md) for the full design.

## Key Design Rules

1. Python-first. Bash only in `bin/benchmark`.
2. Wraps `vllm serve` and `vllm bench serve` as subprocesses.
3. Every run writes explicit `run_config.json`. No metadata in paths.
4. Server logs captured to disk. Health-check failures include tailed logs.
5. Concrete before abstract. No backend interfaces until 2+ backends exist.

## Modules

- `cli.py` — Click CLI (sweep, run, plot, compare, logs)
- `config.py` — 4-layer config resolution
- `runner.py` — Sweep orchestration, single-run pipeline
- `serve.py` — vLLM server lifecycle (start, health-check, stop)
- `metrics.py` — vllm output parsing, statistics, result serialization
- `plot.py` — Matplotlib/seaborn plot generation
- `utils.py` — GPU info, file ops, timing
```

**Step 3: Write docs/USAGE.md**

```markdown
# Usage Guide

## Running Benchmarks

### Sweep
```bash
benchmark sweep --model meta-llama/Llama-3.1-8B-Instruct \
    --rates 1 2 4 8 16 \
    --concurrencies 1 4 16 64 128 \
    --dataset sharegpt \
    --dataset-path /data/sharegpt.json \
    --num-prompts 1000
```

### Single Run
```bash
benchmark run --model meta-llama/Llama-3.1-8B-Instruct \
    --rate 8 --concurrency 32
```

## Resuming Failed Sweeps

The sweep command skips combinations that already have `run_metrics.json`. If a sweep fails partway through, re-run the same command to resume from the last completed combination.

## Results Structure

```
results/<model-safe-name>/sweeps/
├── rate_8.0_conc_1/
│   ├── run_config.json
│   ├── run_metrics.json
│   └── server.log
├── rate_8.0_conc_4/
│   └── ...
├── rate_8.0_conc_16/
│   └── ...
└── plots/
    ├── throughput_vs_ttft.png
    ├── concurrency_scaling_rate_8.0.png
    └── tokens_per_user.png
```

## Datasets

- `random` — synthetic random prompts (uses `--input-len` and `--output-len`)
- `sharegpt` — ShareGPT conversations (requires `--dataset-path`)
- `sonnet` — Sonnet dataset (requires `--dataset-path`)
- Custom path — any JSON dataset file
```

**Step 4: Commit**

```bash
git add README.md docs/ARCHITECTURE.md docs/USAGE.md
git commit -m "docs: add README, architecture, and usage documentation"
```

---

### Task 11: Final Verification

**Step 1: Run full test suite**

```bash
python -m pytest lm_benchmarks/ -v
```
Expected: All ~40 tests PASS.

**Step 2: Verify CLI help text**

```bash
python -m lm_benchmarks.cli --help
python -m lm_benchmarks.cli sweep --help
python -m lm_benchmarks.cli run --help
```
Expected: Help text for all subcommands with correct option names.

**Step 3: Verify bin/benchmark wrapper**

```bash
./bin/benchmark --help
```
Expected: Same help output as `python -m lm_benchmarks.cli --help`.

**Step 4: Install in dev mode**

```bash
pip install -e ".[dev]"
benchmark --help
```
Expected: CLI works via installed entry point.

**Step 5: Commit if any fixes needed**

```bash
git add -A
git commit -m "chore: final verification and fixes"
```

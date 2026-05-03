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

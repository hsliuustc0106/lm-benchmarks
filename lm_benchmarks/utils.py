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

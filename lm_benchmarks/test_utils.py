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

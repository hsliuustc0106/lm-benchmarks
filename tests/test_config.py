import os
import tempfile
from pathlib import Path
import lm_benchmarks.config as config


def test_defaults_are_set():
    """Built-in defaults exist for all required keys."""
    cfg = config.load()
    assert cfg["engine"] == "vllm"
    assert cfg["port"] == 8080
    assert cfg["server_timeout"] == 900
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
    assert cfg["num_prompts"] == 999
    assert isinstance(cfg["num_prompts"], int)
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

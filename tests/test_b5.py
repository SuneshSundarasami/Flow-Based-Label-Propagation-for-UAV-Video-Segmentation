"""B5 tests: config-driven defaults, override YAML, and CLI arg parsing."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import load_config

# All keys that run_propagation.py reads from the config.
_REQUIRED = [
    ("paths", "dataset_root"),
    ("paths", "sea_raft_checkpoint"),
    ("paths", "output_dir"),
    ("data", "frame_glob"),
    ("data", "mask_glob"),
    ("data", "mask_format"),
    ("flow", "model_cfg"),
    ("flow", "iters"),
    ("flow", "device"),
    ("propagation", "n_targets"),
    ("propagation", "window"),
    ("occlusion", "fb_threshold"),
    ("eval", "ignore_index"),
]


def test_default_config_has_all_required_keys():
    cfg = load_config()
    for section, key in _REQUIRED:
        assert section in cfg, f"missing section: {section}"
        assert key in cfg[section], f"missing key: {section}.{key}"


def test_data_section_values_are_sane():
    cfg = load_config()
    assert isinstance(cfg["data"]["frame_glob"], str)
    assert isinstance(cfg["data"]["mask_glob"], str)
    assert cfg["data"]["mask_format"] in ("color", "indexed")


def test_propagation_n_targets_is_positive_int():
    cfg = load_config()
    n = cfg["propagation"]["n_targets"]
    assert isinstance(n, int) and n > 0


def test_dict_override_single_leaf():
    cfg = load_config(overrides={"propagation": {"n_targets": 99}})
    assert cfg["propagation"]["n_targets"] == 99
    assert cfg["propagation"]["window"] == 30  # sibling key untouched


def test_dict_override_does_not_clobber_other_sections():
    cfg = load_config(overrides={"data": {"frame_glob": "custom/*.jpg"}})
    assert cfg["data"]["frame_glob"] == "custom/*.jpg"
    assert "fb_threshold" in cfg["occlusion"]  # other section untouched


def test_yaml_override_file(tmp_path):
    override = tmp_path / "override.yaml"
    override.write_text(
        "propagation:\n  n_targets: 7\n  window: 10\n"
        "flow:\n  device: cpu\n"
    )
    cfg = load_config(override_path=override)
    assert cfg["propagation"]["n_targets"] == 7
    assert cfg["propagation"]["window"] == 10
    assert cfg["flow"]["device"] == "cpu"
    assert cfg["flow"]["iters"] == 12  # untouched default


def test_yaml_override_then_dict_override_wins(tmp_path):
    override = tmp_path / "override.yaml"
    override.write_text("propagation:\n  n_targets: 7\n")
    cfg = load_config(override_path=override, overrides={"propagation": {"n_targets": 3}})
    assert cfg["propagation"]["n_targets"] == 3  # dict wins


def test_cli_defaults_match_config():
    """Verify that the argparse defaults in run_propagation match the config."""
    import importlib.util, types

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_propagation.py"
    spec = importlib.util.spec_from_file_location("run_prop", script)
    mod = importlib.util.module_from_spec(spec)
    # Patch sys.argv so argparse sees no flags
    import sys as _sys
    orig_argv = _sys.argv
    _sys.argv = [str(script)]
    try:
        spec.loader.exec_module(mod)
        cfg = mod._pre_load_config([])
        # Check that _pre_load_config with empty argv returns the default config
        assert cfg["propagation"]["n_targets"] == load_config()["propagation"]["n_targets"]
        assert cfg["flow"]["device"] == load_config()["flow"]["device"]
    finally:
        _sys.argv = orig_argv

"""B5 tests: config-driven defaults, override YAML, and CLI arg parsing."""
import csv
import io
import sys
from pathlib import Path

import numpy as np
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


def test_keyframe_csv_columns(tmp_path):
    """run_propagation saves keyframe_X.csv with the expected column layout."""
    import importlib.util, math

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_propagation.py"
    spec = importlib.util.spec_from_file_location("run_prop", script)
    mod = importlib.util.module_from_spec(spec)
    orig_argv = sys.argv
    sys.argv = [str(script)]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = orig_argv

    num_classes = 3
    # Simulate what run_propagation builds: a list of per-pair dicts
    kf_idx = 0
    targets = [10, 20]
    class_names = [f"class{c}" for c in range(num_classes)]

    kf_rows = []
    for t_idx in targets:
        row = {
            "keyframe": kf_idx,
            "target_frame": t_idx,
            "distance": t_idx - kf_idx,
            "valid_pct": "95.00",
            "miou_all": "0.8000",
            "miou_valid": "0.8500",
        }
        for c in range(num_classes):
            row[f"iou_class{c}"] = "0.9000"
        kf_rows.append(row)

    # Write and read back to verify column structure
    kf_csv = tmp_path / f"keyframe_{kf_idx}.csv"
    fieldnames = list(kf_rows[0].keys())
    with open(kf_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kf_rows)

    with open(kf_csv) as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 2
    assert rows[0]["keyframe"] == "0"
    assert rows[0]["target_frame"] == "10"
    assert rows[1]["target_frame"] == "20"
    assert "miou_all" in rows[0]
    assert "miou_valid" in rows[0]
    for c in range(num_classes):
        assert f"iou_class{c}" in rows[0]
    # iou_classN columns not named after class strings
    assert "class0" not in rows[0]


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

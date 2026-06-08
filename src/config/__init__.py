"""Configuration loading: a default YAML deep-merged with optional overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

_DEFAULT_PATH = Path(__file__).with_name("default.yaml")


def _deep_merge(base: dict, override: Mapping[str, Any]) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(override_path: str | Path | None = None,
                overrides: Mapping[str, Any] | None = None) -> dict:
    """Load the default config, optionally merged with a YAML file and/or dict.

    Later sources win. Returns a plain nested dict.
    """
    with open(_DEFAULT_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if override_path is not None:
        with open(override_path, "r", encoding="utf-8") as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg

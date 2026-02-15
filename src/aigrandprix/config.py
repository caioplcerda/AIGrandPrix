"""YAML configuration loading and safe nested access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load YAML config, defaulting to configs/default.yaml."""
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        return {}
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def get_section(config: dict, *keys: str, default: Any = None) -> Any:
    """Safe nested dict access: get_section(cfg, 'control', 'pid') -> cfg['control']['pid']."""
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current

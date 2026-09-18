"""YAML configuration loaders for experimental pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    """Returns absolute path to the project root directory."""
    return PROJECT_ROOT


def load_yaml_config(relative_or_abs_path: str | Path) -> Dict[str, Any]:
    """Loads a YAML configuration file safely.

    Args:
        relative_or_abs_path: Relative path from project root or absolute path.

    Returns:
        Dictionary of configuration parameters.
    """
    path = Path(relative_or_abs_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_m4_config() -> Dict[str, Any]:
    return load_yaml_config("experiments/configs/m4_retrieval.yaml")


def load_m5_config() -> Dict[str, Any]:
    return load_yaml_config("experiments/configs/m5_change_detection.yaml")


def load_m6_config() -> Dict[str, Any]:
    return load_yaml_config("experiments/configs/m6_false_alarm.yaml")


def load_m7_config() -> Dict[str, Any]:
    return load_yaml_config("experiments/configs/m7_learned_cd.yaml")

"""Portable, human-readable YAML serialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML


def to_builtin(value):
    """Recursively convert NumPy objects to Python built-ins for clean YAML serialization."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {to_builtin(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_yaml(data, path: Path | str) -> None:
    """Write data to a YAML file, ensuring all types are basic Python types."""
    clean_data = to_builtin(data)
    yaml = YAML()
    yaml.default_flow_style = False

    with Path(path).open("w") as f:
        yaml.dump(clean_data, f)

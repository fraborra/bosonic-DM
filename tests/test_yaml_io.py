# Copyright (C) 2025 Francesco Borra
#

from __future__ import annotations

from pathlib import Path

import numpy as np
from dbetto import Props

from bosonic_dm.yaml_io import migrate_yaml_file, read_yaml, write_yaml


def test_write_yaml_uses_portable_types(tmp_path: Path) -> None:
    output = tmp_path / "portable.yaml"
    write_yaml(
        {
            "float": np.float64(1.25),
            "integer": np.int64(4),
            "array": np.array([1, 2]),
            "missing": np.nan,
        },
        output,
    )

    text = output.read_text(encoding="utf-8")
    assert "!!python" not in text
    assert "numpy.core" not in text
    assert read_yaml(output) == {
        "float": 1.25,
        "integer": 4,
        "array": [1, 2],
        "missing": None,
    }


def test_migrate_yaml_file_rewrites_legacy_values(tmp_path: Path) -> None:
    source = tmp_path / "legacy.yaml"
    destination = tmp_path / "migrated.yaml"
    Props.write_to(source, {"value": np.float64(2.5)})

    result = migrate_yaml_file(source, destination)

    assert result == destination
    assert read_yaml(destination) == {"value": 2.5}
    assert "!!python" not in destination.read_text(encoding="utf-8")

# Copyright (C) 2026 Francesco Borra
#

"""Shared plotting constants and helpers."""

from __future__ import annotations

import numpy as np

from bosonic_dm.constants import DET_TYPE_MAP

# Kept under the old name for the plotting modules and notebooks.
_DET_TYPE_MAP = DET_TYPE_MAP
_DET_TYPE_COLOR = {
    "BEGe": "tab:blue",
    "COAX": "tab:orange",
    "ICPC": "tab:green",
    "PPC": "tab:red",
}


def true_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive index ranges of consecutive ``True`` values."""
    padded = np.concatenate([[False], np.asarray(flags, dtype=bool), [False]])
    changes = np.flatnonzero(np.diff(padded.astype(int)))
    return [
        (int(a), int(b) - 1) for a, b in zip(changes[::2], changes[1::2], strict=True)
    ]

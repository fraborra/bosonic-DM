# Copyright (C) 2025 Francesco Borra
#

"""Small shared helpers for bosonic-DM."""

from __future__ import annotations

import numpy as np


def expand_range(item):
    if ".." not in item:
        return [item]  # already single
    start, end = item.split("..")

    prefix = start[0]  # 'r'
    s = int(start[1:])
    e = int(end[1:])
    width = len(start) - 1  # number of digits

    return [f"{prefix}{i:0{width}d}" for i in range(s, e + 1)]


def clean_array(arr):
    arr = np.asarray(arr)
    arr = arr.astype(float)
    return arr[(arr != 0) & np.isfinite(arr)]


def select_channel(energies, channels, rawid):
    return energies[channels == rawid]

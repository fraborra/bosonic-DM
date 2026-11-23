# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure

from bosonic_dm.plotting.sensitivity import plot_stan_grid_comparison


def _tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    nan = float("nan")
    comparison = pl.DataFrame(
        {
            "mass_keV": [100.0, 200.0, 300.0],
            "near_masked_line": [False, True, False],
            "n_toys": [10, 10, 10],
            "n_failed": [0, 1, 10],
            "median_ratio": [1.01, 0.97, nan],
            "median_ratio_err": [0.02, 0.03, nan],
        }
    )
    rng = np.random.default_rng(1)
    rows = []
    for m in (100.0, 200.0, 300.0):
        for t in range(10):
            r90 = 1.0 + 0.1 * t
            rows.append((m, t, r90, r90 + rng.normal(0, 0.01), 0.01, True))
    rows[10] = (200.0, 0, 1.0, nan, nan, False)
    per_toy = pl.DataFrame(
        rows,
        schema={
            "mass_keV": pl.Float64,
            "toy": pl.Int64,
            "r90": pl.Float64,
            "r90_stan": pl.Float64,
            "r90_stan_mcse": pl.Float64,
            "passed": pl.Boolean,
        },
        orient="row",
    )
    return comparison, per_toy


def test_plot_stan_grid_comparison(tmp_path: Path) -> None:
    comparison, per_toy = _tables()
    out = tmp_path / "cmp.png"
    fig = plot_stan_grid_comparison(comparison, per_toy, title="t", output_path=out)
    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
        assert out.exists()
        assert out.stat().st_size > 0
    finally:
        plt.close(fig)

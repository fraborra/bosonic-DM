# Copyright (C) 2026 Francesco Borra
#

"""Plot of the expected sensitivity: limit bands over the mass range."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bosonic_dm.plotting.signal_fit import MASKED_COLOR
from bosonic_dm.plotting.utils import true_runs

BAND_95_COLOR = "#ffcc00"
BAND_68_COLOR = "#00b140"
RATE_LABEL = r"$R_{90}$ [counts / (kg yr)]"


def plot_expected_limits(
    summary: pl.DataFrame,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    ylabel: str = RATE_LABEL,
    output_path: str | Path | None = None,
) -> Figure:
    """Brazil plot of the expected upper limit versus DM mass.

    Draws the 95% and 68% central bands and the median of the limit over the
    pseudo-experiments, the Asimov limit, and grey spans where the fit window
    contains bins masked around gamma lines.

    Parameters
    ----------
    summary
        ``ExpectedLimits.summary``.
    ax
        Axes to draw on; a new figure is created if omitted.
    title
        Axes title.
    ylabel
        Y-axis label.
    output_path
        Optional file to save the figure to.
    """
    table = summary.sort("mass_keV")
    mass = table["mass_keV"].to_numpy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    if "near_masked_line" in table.columns and len(mass) > 1:
        half_step = 0.5 * float(np.median(np.diff(mass)))
        runs = true_runs(table["near_masked_line"].to_numpy())
        for index, (first, last) in enumerate(runs):
            ax.axvspan(
                mass[first] - half_step,
                mass[last] + half_step,
                color=MASKED_COLOR,
                lw=0,
                zorder=0,
                label="window near masked line" if index == 0 else None,
            )

    ax.fill_between(
        mass,
        table["r90_q025"].to_numpy(),
        table["r90_q975"].to_numpy(),
        color=BAND_95_COLOR,
        lw=0,
        label="95% expected",
    )
    ax.fill_between(
        mass,
        table["r90_q16"].to_numpy(),
        table["r90_q84"].to_numpy(),
        color=BAND_68_COLOR,
        lw=0,
        label="68% expected",
    )
    ax.plot(mass, table["r90_median"].to_numpy(), "k--", lw=1.2, label="median")
    if "r90_asimov" in table.columns:
        ax.plot(
            mass,
            table["r90_asimov"].to_numpy(),
            ":",
            color="0.3",
            lw=1.0,
            label="Asimov",
        )
    ax.set_yscale("log")
    ax.set_xlim(mass.min(), mass.max())
    ax.set_xlabel("DM mass [keV]")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150)
    return fig

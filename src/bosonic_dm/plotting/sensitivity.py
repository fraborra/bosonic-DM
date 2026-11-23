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
from scipy.stats import norm

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


def plot_stan_grid_comparison(
    comparison: pl.DataFrame,
    per_toy: pl.DataFrame,
    *,
    title: str | None = None,
    output_path: str | Path | None = None,
) -> Figure:
    """Compare the Stan and grid-posterior 90% upper limits.

    The top panel shows the median ratio of the Stan to the grid limit versus
    DM mass. The bottom panel shows the pooled pulls
    ``(r90_stan - r90) / r90_stan_mcse`` of the passed toys, with the
    standard normal and a width-1.5 normal density overlaid.

    Parameters
    ----------
    comparison
        One row per mass with ``mass_keV``, ``near_masked_line``,
        ``median_ratio`` and ``median_ratio_err``. Rows with a NaN ratio are
        skipped.
    per_toy
        One row per toy with ``mass_keV``, ``r90``, ``r90_stan``,
        ``r90_stan_mcse`` and ``passed``.
    title
        Optional figure title.
    output_path
        If given, the figure is saved to this path.

    Returns
    -------
    Figure
        The figure with two axes (ratio versus mass, pull histogram).
    """
    fig, (ax_ratio, ax_pull) = plt.subplots(2, 1, figsize=(7.0, 8.0))

    mass = comparison["mass_keV"].to_numpy().astype(float)
    ratio = comparison["median_ratio"].to_numpy().astype(float)
    err = comparison["median_ratio_err"].to_numpy().astype(float)
    near = comparison["near_masked_line"].to_numpy().astype(bool)
    valid = np.isfinite(mass) & np.isfinite(ratio)
    err = np.where(np.isfinite(err), err, 0.0)
    for flagged, color, marker, label in (
        (False, "C0", "o", "away from masked lines"),
        (True, "C3", "s", "near masked line"),
    ):
        sel = valid & (near == flagged)
        if sel.any():
            ax_ratio.errorbar(
                mass[sel],
                ratio[sel],
                yerr=err[sel],
                fmt=marker,
                color=color,
                ms=4,
                capsize=2,
                label=label,
            )
    ax_ratio.axhline(1.0, color="k", ls="--", lw=1.0)
    ax_ratio.set_xlabel("DM mass [keV]")
    ax_ratio.set_ylabel("median R90 Stan / grid")
    if ax_ratio.get_legend_handles_labels()[0]:
        ax_ratio.legend(loc="best", frameon=False)
    if title:
        ax_ratio.set_title(title)

    passed = per_toy["passed"].to_numpy().astype(bool)
    stan = per_toy["r90_stan"].to_numpy().astype(float)
    grid = per_toy["r90"].to_numpy().astype(float)
    mcse = per_toy["r90_stan_mcse"].to_numpy().astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pull = (stan - grid) / mcse
    pull = pull[passed & np.isfinite(pull)]
    if pull.size:
        ax_pull.hist(
            pull,
            bins=40,
            density=True,
            color="0.75",
            label=rf"pulls: mean {pull.mean():.2f}, sd "
            rf"{pull.std(ddof=1) if pull.size > 1 else float('nan'):.2f}",
        )
        lim = max(4.0, float(np.abs(pull).max()))
    else:
        lim = 4.0
    z = np.linspace(-lim, lim, 400)
    ax_pull.plot(z, norm.pdf(z), "k-", lw=1.2, label=r"$N(0, 1)$")
    ax_pull.plot(z, norm.pdf(z, scale=1.5), "C1--", lw=1.2, label=r"$N(0, 1.5^2)$")
    ax_pull.set_xlabel(r"pull $(R_{90}^{\rm Stan} - R_{90}^{\rm grid})$ / MCSE")
    ax_pull.set_ylabel("density")
    ax_pull.legend(loc="best", frameon=False)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150)
    return fig

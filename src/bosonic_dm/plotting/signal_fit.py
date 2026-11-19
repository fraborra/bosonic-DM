# Copyright (C) 2026 Francesco Borra
#

"""Plot of the fixed-background signal fit in one mass window."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from bosonic_dm.fitting.signal_template import SignalWindow
from bosonic_dm.plotting.utils import true_runs

MASKED_COLOR = "0.85"


def plot_window_fit(
    window: SignalWindow,
    rate: float,
    *,
    counts: np.ndarray | None = None,
    counts_label: str = "counts",
    rate_label: str = r"$\hat B + R_{90} S$",
    title: str | None = None,
    output_path: str | Path | None = None,
) -> Figure:
    """Counts, fixed background and background plus signal in one fit window.

    One panel per group. Bins left out of the likelihood are shaded.

    Parameters
    ----------
    window
        Fit window from ``build_signal_window``. Its counts are drawn unless
        ``counts`` is given; pass a window with pseudo-data
        (``SignalWindow.with_counts``) to keep the observed counts blinded.
    rate
        Signal rate to draw on top of ``B_hat``, in counts/(kg yr), e.g. the
        median expected limit.
    counts
        Counts to draw, shape ``(groups, bins)``; defaults to the window
        counts.
    counts_label
        Legend label of the counts, e.g. ``"toy"``.
    rate_label
        Legend label of ``B_hat + rate * S``.
    title
        Figure title; defaults to the mass.
    output_path
        Optional file to save the figure to.
    """
    edges = window.bin_edges
    observed = window.stacked("counts") if counts is None else np.asarray(counts)
    b_hat = window.stacked("b_hat")
    template = window.stacked("signal_template")
    use_bin = window.stacked("use_bin").astype(bool)
    if observed.shape != b_hat.shape:
        msg = f"counts must have shape {b_hat.shape}; got {observed.shape}."
        raise ValueError(msg)

    n_groups = len(window.groups)
    fig, axes = plt.subplots(
        n_groups, 1, figsize=(7, 1.9 * n_groups + 0.6), sharex=True, squeeze=False
    )
    for row, (ax, name) in enumerate(zip(axes[:, 0], window.group_names, strict=True)):
        for first, last in true_runs(~use_bin[row]):
            ax.axvspan(edges[first], edges[last + 1], color=MASKED_COLOR, lw=0)
        ax.stairs(
            observed[row], edges, baseline=None, color="k", lw=1.0, label=counts_label
        )
        ax.stairs(
            b_hat[row], edges, baseline=None, color="C0", lw=1.5, label=r"$\hat B$"
        )
        ax.stairs(
            b_hat[row] + rate * template[row],
            edges,
            baseline=None,
            color="C3",
            lw=1.5,
            label=rate_label,
        )
        ax.text(
            0.02,
            0.85,
            name,
            transform=ax.transAxes,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
        ax.set_ylabel("counts / bin")
    axes[0, 0].legend(loc="upper right", frameon=False, fontsize="small")
    axes[-1, 0].set_xlabel("energy [keV]")
    axes[-1, 0].set_xlim(edges[0], edges[-1])
    fig.suptitle(title or f"m = {window.mass_keV:g} keV, R = {rate:.3g} / (kg yr)")
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150)
    return fig

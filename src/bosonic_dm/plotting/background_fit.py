# Copyright (C) 2026 Francesco Borra
#

"""Residual, chain and corner plots for the Stan background fit.

Residuals compare the observed counts ``n`` with ``B_hat`` (posterior median
of ``expected_counts``)::

    r = (n - B_hat) / sqrt(B_hat + Var_post[expected_counts])

The posterior variance comes from the thinned draws kept in the fit summary,
so ``r`` is a predictive pull: approximately standard normal where the model
describes the data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure

from bosonic_dm.fitting.background_fit import (
    BACKGROUND_PARAMETERS,
    BackgroundFitSummary,
)

PARAMETER_LABELS = {
    "N_2nu": r"$N_{2\nu}$",
    "N_ar39": r"$N_{\mathrm{Ar39}}$",
    "c0": r"$c_0$",
    "cQ": r"$c_Q$",
    "mu": r"$\mu$",
    "sigma": r"$\sigma$",
    "tau": r"$\tau$",
    "base_alpha": r"$\alpha_G$",
    "base_beta": r"$\beta_G$",
}

_COMPONENT_STYLE = {
    "expected_2nu": ("C1", r"$2\nu\beta\beta$"),
    "expected_linear": ("C2", "linear"),
    "expected_ar39": ("C0", r"$^{39}$Ar"),
}


def normalized_residuals(
    summary: BackgroundFitSummary, counts: Sequence[int] | np.ndarray
) -> np.ndarray:
    """Per-bin predictive pulls of ``counts`` against ``summary.b_hat``."""
    n = np.asarray(counts, dtype=float)
    variance = summary.b_hat + summary.expected_counts_draws.var(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(variance > 0, (n - summary.b_hat) / np.sqrt(variance), np.nan)


def _excluded_windows(used: np.ndarray) -> list[np.ndarray]:
    """Bin indices of each run of consecutive excluded bins."""
    excluded = np.flatnonzero(~used)
    if excluded.size == 0:
        return []
    return np.split(excluded, np.flatnonzero(np.diff(excluded) > 1) + 1)


def excluded_window_excess(
    summary: BackgroundFitSummary, counts: Sequence[int] | np.ndarray
) -> pl.DataFrame:
    """Observed minus predicted counts in each window left out of the fit.

    ``significance`` is the excess over
    ``sqrt(expected + Var_post[expected])``, summed over the window. A window
    whose significance is compatible with zero holds no visible peak, so its
    entry in the peak list can be dropped and the bins returned to the fit.
    """
    n = np.asarray(counts, dtype=float)
    edges = summary.bin_edges
    rows = []
    for idx in _excluded_windows(summary.used_in_fit):
        observed = n[idx].sum()
        expected = summary.b_hat[idx].sum()
        posterior_var = summary.expected_counts_draws[:, idx].sum(axis=1).var()
        rows.append(
            {
                "lo_keV": float(edges[idx[0]]),
                "hi_keV": float(edges[idx[-1] + 1]),
                "centre_keV": float(0.5 * (edges[idx[0]] + edges[idx[-1] + 1])),
                "observed": observed,
                "expected": expected,
                "excess": observed - expected,
                "significance": (observed - expected)
                / np.sqrt(expected + posterior_var),
            }
        )
    schema = {
        "lo_keV": pl.Float64,
        "hi_keV": pl.Float64,
        "centre_keV": pl.Float64,
        "observed": pl.Float64,
        "expected": pl.Float64,
        "excess": pl.Float64,
        "significance": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def plot_background_fit_residuals(
    summary: BackgroundFitSummary,
    counts: Sequence[int] | np.ndarray,
    *,
    include_excluded: bool = False,
    title: str | None = None,
    yscale: str = "log",
    residual_limit: float | None = 5.0,
    output_path: Path | None = None,
) -> Figure:
    """Spectrum with the fitted model and a normalized-residual panel.

    Parameters
    ----------
    summary
        Output of ``summarize_background_fit``.
    counts
        Observed counts per bin, e.g. ``stan_data["counts"]``.
    include_excluded
        If ``False``, only bins that entered the likelihood get residuals;
        excluded windows are shaded and left empty. If ``True``, excluded
        bins are drawn too, in a separate colour, and each window is labelled
        with its summed significance (see :func:`excluded_window_excess`).
    title
        Figure title, e.g. the detector group.
    yscale
        Scale of the spectrum axis.
    residual_limit
        Symmetric y-limit of the residual panel; points beyond it are drawn
        as triangles on the edge. Automatic (no clipping) if ``None``.
    output_path
        If given, the figure is also saved there.

    Returns
    -------
    Figure
        The figure, left open so it can be shown or edited.
    """
    n = np.asarray(counts, dtype=float)
    edges = summary.bin_edges
    centres = 0.5 * (edges[:-1] + edges[1:])
    used = summary.used_in_fit
    pulls = normalized_residuals(summary, n)

    fig, (ax_spec, ax_res) = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.06},
    )

    ax_spec.stairs(n, edges, color="black", lw=0.8, label="data")
    for name, values in summary.component_medians.items():
        color, label = _COMPONENT_STYLE.get(name, (None, name))
        ax_spec.stairs(values, edges, color=color, lw=1.0, label=label)
    ax_spec.stairs(summary.b_hat, edges, color="C3", lw=1.4, label=r"$\hat{B}$")
    for idx in _excluded_windows(used):
        for ax in (ax_spec, ax_res):
            ax.axvspan(edges[idx[0]], edges[idx[-1] + 1], color="grey", alpha=0.2, lw=0)
    ax_spec.set(ylabel="Counts / bin", yscale=yscale)
    ax_spec.set_ylim(10**-2, np.max(values) * 1.3)
    ax_spec.legend(loc="upper right", fontsize=9)

    for k, color in ((2, "#f2e6a0"), (1, "#c9dfa0")):
        ax_res.axhspan(-k, k, color=color, lw=0, zorder=0)
    ax_res.axhline(0.0, color="grey", lw=0.7)

    def draw(mask: np.ndarray, color: str, label: str) -> None:
        y = pulls[mask]
        if residual_limit is None:
            ax_res.plot(centres[mask], y, ".", ms=2.5, color=color, label=label)
            return
        inside = np.abs(y) <= residual_limit
        ax_res.plot(
            centres[mask][inside], y[inside], ".", ms=2.5, color=color, label=label
        )
        for sign, marker in ((1, "^"), (-1, "v")):
            out = sign * y > residual_limit
            ax_res.plot(
                centres[mask][out],
                np.full(out.sum(), sign * 0.92 * residual_limit),
                marker,
                ms=3.5,
                color=color,
                ls="none",
            )

    draw(used, "black", "fitted bins")

    if include_excluded and (~used).any():
        draw(~used, "C3", "excluded bins")
        table = excluded_window_excess(summary, n)
        for row in table.iter_rows(named=True):
            ax_spec.annotate(
                f"{row['significance']:+.1f}$\\sigma$",
                xy=(row["centre_keV"], 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, -4),
                textcoords="offset points",
                ha="center",
                va="top",
                rotation=90,
                fontsize=7,
                color="C3",
            )
        ax_res.legend(loc="upper right", fontsize=8, ncols=2)

    if residual_limit is not None:
        ax_res.set_ylim(-residual_limit, residual_limit)
    ax_res.set(
        xlabel="Energy [keV]",
        ylabel=r"$(n - \hat{B}) / \sigma$",
        xlim=(edges[0], edges[-1]),
    )
    suffix = "all bins" if include_excluded else "fitted bins"
    ax_spec.set_title(f"{title} — {suffix}" if title else suffix)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def posterior_draws(
    posterior: Any, var_names: Sequence[str] = BACKGROUND_PARAMETERS
) -> np.ndarray:
    """Draws of scalar parameters, shaped ``(chains, draws, parameters)``.

    ``posterior`` is a ``cmdstanpy.CmdStanMCMC`` or an ArviZ object with a
    ``posterior`` group (e.g. ``to_inference_data(...)`` or a reloaded
    ``posterior.nc``).
    """
    if hasattr(posterior, "draws") and hasattr(posterior, "column_names"):
        draws = posterior.draws(concat_chains=False)  # (draws, chains, columns)
        columns = list(posterior.column_names)
        idx = [columns.index(name) for name in var_names]
        return np.moveaxis(draws[:, :, idx], 0, 1)
    group = posterior["posterior"]
    return np.stack([np.asarray(group[name].values) for name in var_names], axis=-1)


def plot_background_chains(
    posterior: Any,
    var_names: Sequence[str] = BACKGROUND_PARAMETERS,
    *,
    title: str | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Trace (left) and per-chain marginal (right) for each parameter.

    Well-mixed chains overlap as stationary noise on the left and give
    matching histograms on the right.
    """
    draws = posterior_draws(posterior, var_names)
    n_chains = draws.shape[0]
    fig, axes = plt.subplots(
        len(var_names),
        2,
        figsize=(11, 1.6 * len(var_names)),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.04},
        squeeze=False,
        layout="constrained",
    )
    for k, name in enumerate(var_names):
        ax_trace, ax_hist = axes[k]
        bins = np.histogram_bin_edges(draws[:, :, k], bins=40)
        for chain in range(n_chains):
            color = f"C{chain % 10}"
            ax_trace.plot(draws[chain, :, k], color=color, lw=0.4, alpha=0.8)
            ax_hist.hist(
                draws[chain, :, k],
                bins=bins,
                histtype="step",
                color=color,
                orientation="horizontal",
                density=True,
            )
        ax_trace.set_ylabel(PARAMETER_LABELS.get(name, name))
        ax_trace.set_xlim(0, draws.shape[1] - 1)
        ax_hist.sharey(ax_trace)
        ax_hist.tick_params(
            labelleft=False, left=False, labelbottom=False, bottom=False
        )
        if k < len(var_names) - 1:
            ax_trace.tick_params(labelbottom=False)
    axes[-1, 0].set_xlabel("Draw")
    if title:
        fig.suptitle(title)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_background_corner(
    posterior: Any,
    var_names: Sequence[str] = BACKGROUND_PARAMETERS,
    *,
    title: str | None = None,
    output_path: Path | None = None,
    corner_kwargs: Mapping[str, Any] | None = None,
) -> Figure:
    """Corner plot of the joint posterior, all chains pooled.

    Titles show the median and the 16th-84th percentile range;
    ``corner_kwargs`` is passed on to ``corner.corner`` and overrides these
    defaults.
    """
    import corner  # noqa: PLC0415

    samples = posterior_draws(posterior, var_names).reshape(-1, len(var_names))
    options = {
        "labels": [PARAMETER_LABELS.get(name, name) for name in var_names],
        "quantiles": (0.16, 0.5, 0.84),
        "show_titles": True,
        "title_fmt": ".3g",
        "title_kwargs": {"fontsize": 9},
        "label_kwargs": {"fontsize": 11},
        "smooth": 1.0,
        **(corner_kwargs or {}),
    }
    fig = corner.corner(samples, **options)
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig

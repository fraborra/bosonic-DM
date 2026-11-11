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
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

from bosonic_dm.fitting.background_fit import (
    BACKGROUND_PARAMETERS,
    BackgroundFitSummary,
    sample_prior_predictive,
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


def _smooth_curve(
    edges: np.ndarray, values: np.ndarray, points_per_bin: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth line through per-bin values, on a grid finer than the bins.

    Monotone cubic (PCHIP) interpolation of ``log(values)`` between bin
    centres, so the line passes through every bin's value, cannot overshoot,
    and follows steep falls on a log axis. It reaches half a bin beyond the
    first and last positive bins and is ``nan`` where the values are zero
    (e.g. the Ar39 component above its endpoint).
    """
    centres = 0.5 * (edges[:-1] + edges[1:])
    grid = np.linspace(edges[0], edges[-1], points_per_bin * len(centres) + 1)
    curve = np.full_like(grid, np.nan)
    positive = np.flatnonzero(values > 0)
    if positive.size < 2:
        return grid, np.interp(grid, centres, values)
    first, last = positive[0], positive[-1]
    inside = (grid >= edges[first]) & (grid <= edges[last + 1])
    interpolator = PchipInterpolator(
        centres[positive], np.log(values[positive]), extrapolate=True
    )
    curve[inside] = np.exp(interpolator(grid[inside]))
    return grid, curve


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


def _bin_draws(posterior: Any, name: str = "expected_counts") -> np.ndarray:
    """All draws of a per-bin variable, chains pooled, shaped ``(draws, n_bins)``."""
    if hasattr(posterior, "stan_variable"):
        return np.asarray(posterior.stan_variable(name), dtype=float)
    values = np.asarray(posterior["posterior"][name].values, dtype=float)
    return values.reshape(-1, values.shape[-1])


def posterior_band(posterior: Any, prob: float = 0.68) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin equal-tailed ``prob`` interval of ``expected_counts``.

    ``posterior`` is a ``cmdstanpy.CmdStanMCMC`` or an ArviZ object with a
    ``posterior`` group (e.g. a reloaded ``posterior.nc``). Every posterior
    draw is used, not the thinned ``expected_counts_draws`` of the summary:
    with 200 draws a 95% interval rests on about five draws per tail.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Lower and upper edges, one value per bin.
    """
    return _quantile_band(_bin_draws(posterior), prob)


def _quantile_band(draws: np.ndarray, prob: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < prob < 1:
        msg = f"prob must lie in (0, 1), got {prob}."
        raise ValueError(msg)
    lo, hi = np.quantile(draws, [(1 - prob) / 2, (1 + prob) / 2], axis=0)
    return lo, hi


def plot_background_fit_residuals(
    summary: BackgroundFitSummary,
    counts: Sequence[int] | np.ndarray,
    *,
    include_excluded: bool = False,
    title: str | None = None,
    yscale: str = "log",
    residual_limit: float | None = 5.0,
    points_per_bin: int = 5,
    posterior: Any | None = None,
    band_probs: Sequence[float] = (0.68, 0.95),
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
    points_per_bin
        Points per bin of the smooth model lines, which interpolate the
        per-bin ``B_hat`` and component medians (see ``_smooth_curve``).
    posterior
        Full posterior of the same fit, a ``cmdstanpy.CmdStanMCMC`` or an
        ArviZ object such as ``az.from_netcdf(".../posterior.nc")``. If given,
        the spectrum panel shows the posterior bands of ``expected_counts``
        around ``B_hat`` (see :func:`posterior_band`), computed from every
        draw. If ``None``, no band is drawn.
    band_probs
        Probabilities of the bands drawn when ``posterior`` is given.
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

    fig = plt.figure(figsize=(12, 7))
    grid = fig.add_gridspec(
        2, 2, height_ratios=[3, 1.3], width_ratios=[7, 1], hspace=0.06, wspace=0.03
    )
    ax_spec = fig.add_subplot(grid[0, 0])
    ax_res = fig.add_subplot(grid[1, 0], sharex=ax_spec)
    ax_hist = fig.add_subplot(grid[1, 1], sharey=ax_res)
    ax_spec.tick_params(labelbottom=False)

    for name, values in summary.component_medians.items():
        color, label = _COMPONENT_STYLE.get(name, (None, name))
        ax_spec.plot(
            *_smooth_curve(edges, values, points_per_bin),
            color=color,
            lw=1.0,
            label=label,
        )
    top = max(float(n.max()), float(summary.b_hat.max()))
    if posterior is not None and len(band_probs):
        draws = _bin_draws(posterior)
        if draws.shape[1] != summary.b_hat.size:
            msg = (
                f"Posterior has {draws.shape[1]} bins, the summary "
                f"{summary.b_hat.size}; pass the posterior of the same fit."
            )
            raise ValueError(msg)
        probs = sorted(band_probs, reverse=True)  # widest first, narrowest on top
        alphas = np.linspace(0.15, 0.35, len(probs)) if len(probs) > 1 else [0.25]
        for prob, alpha in zip(probs, alphas, strict=True):
            lo, hi = _quantile_band(draws, prob)
            grid_x, lo_curve = _smooth_curve(edges, lo, points_per_bin)
            _, hi_curve = _smooth_curve(edges, hi, points_per_bin)
            ax_spec.fill_between(
                grid_x,
                lo_curve,
                hi_curve,
                color="C3",
                alpha=alpha,
                lw=0,
                label=f"{prob:.0%} posterior",
            )
            top = max(top, float(hi.max()))
    ax_spec.plot(
        *_smooth_curve(edges, summary.b_hat, points_per_bin),
        color="C3",
        lw=1.4,
        label=r"$\hat{B}$",
    )
    ax_spec.plot(centres, n, ".", ms=2.5, color="black", label="data", zorder=3)
    for idx in _excluded_windows(used):
        for ax in (ax_spec, ax_res):
            ax.axvspan(edges[idx[0]], edges[idx[-1] + 1], color="grey", alpha=0.2, lw=0)
    ax_spec.set(ylabel="Events / keV", yscale=yscale)
    ax_spec.set_ylim(10**-2, 1.5 * top)
    ax_spec.legend(loc="upper right", fontsize=9)

    for ax in (ax_res, ax_hist):
        ax.axhline(0.0, color="grey", lw=0.7)
        for k, color in ((1, "#8cc152"), (2, "#e6c229")):  # green 1, yellow 2 sigma
            for sign in (1, -1):
                ax.axhline(sign * k, color=color, lw=1.2, ls="--", zorder=0)

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
        ylabel="Normalized residuals",
        xlim=(edges[0], edges[-1]),
    )
    _pull_histogram(
        ax_hist,
        pulls[used],
        pulls[~used] if include_excluded else None,
        residual_limit,
    )
    suffix = "all bins" if include_excluded else "fitted bins"
    ax_spec.set_title(f"{title} — {suffix}" if title else suffix)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def _pull_histogram(
    ax: Any,
    fitted: np.ndarray,
    excluded: np.ndarray | None,
    limit: float | None,
) -> None:
    """Horizontal pull histogram with the standard normal density.

    Pulls beyond ``limit`` are left out of the histogram and counted in the
    text, which also gives the mean and standard deviation of the fitted-bin
    pulls (0 and 1 for a good fit).
    """
    fitted = fitted[np.isfinite(fitted)]
    if limit is None:
        limit = max(3.0, float(np.max(np.abs(fitted))) if fitted.size else 3.0)
    bins = np.linspace(-limit, limit, round(2 * limit / 0.25) + 1)
    ax.hist(
        fitted,
        bins=bins,
        density=True,
        orientation="horizontal",
        histtype="stepfilled",
        color="0.6",
        alpha=0.7,
    )
    if excluded is not None:
        excluded = excluded[np.isfinite(excluded)]
        if excluded.size:
            ax.hist(
                excluded,
                bins=bins,
                density=True,
                orientation="horizontal",
                histtype="step",
                color="C3",
            )
    y = np.linspace(-limit, limit, 300)
    ax.plot(norm.pdf(y), y, color="C0", lw=1.0)
    outside = int(np.sum(np.abs(fitted) > limit))
    text = f"mean {fitted.mean():+.2f}\nsd {fitted.std():.2f}"
    if outside:
        text += f"\n{outside} beyond ±{limit:g}"
    ax.text(0.95, 0.97, text, transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.set_xlabel("density", fontsize=8)
    ax.tick_params(labelleft=False, labelsize=7)
    ax.set_xlim(left=0)


def _draws(posterior: Any, var_names: Sequence[str]) -> np.ndarray:
    if hasattr(posterior, "draws") and hasattr(posterior, "column_names"):
        draws = posterior.draws(concat_chains=False)  # (draws, chains, columns)
        columns = list(posterior.column_names)
        idx = [columns.index(name) for name in var_names]
        return np.moveaxis(draws[:, :, idx], 0, 1)
    group = posterior["posterior"]
    return np.stack([np.asarray(group[name].values) for name in var_names], axis=-1)


def _sampled_parameters(posterior: Any) -> list[str]:
    """Background parameters in ``posterior`` whose draws are not constant.

    Fixed parameters (e.g. ``c0 = cQ = 0`` in the model without the linear
    term) are still written by Stan, but with a constant value.
    """
    if hasattr(posterior, "draws") and hasattr(posterior, "column_names"):
        present = set(posterior.column_names)
    else:
        present = set(posterior["posterior"].data_vars)
    names = [name for name in BACKGROUND_PARAMETERS if name in present]
    draws = _draws(posterior, names)
    return [name for k, name in enumerate(names) if np.ptp(draws[..., k]) > 0]


def posterior_draws(
    posterior: Any, var_names: Sequence[str] | None = None
) -> np.ndarray:
    """Draws of scalar parameters, shaped ``(chains, draws, parameters)``.

    ``posterior`` is a ``cmdstanpy.CmdStanMCMC`` or an ArviZ object with a
    ``posterior`` group (e.g. ``to_inference_data(...)`` or a reloaded
    ``posterior.nc``). ``var_names = None`` selects the sampled background
    parameters, skipping fixed ones.
    """
    if var_names is None:
        var_names = _sampled_parameters(posterior)
    return _draws(posterior, var_names)


def plot_background_chains(
    posterior: Any,
    var_names: Sequence[str] | None = None,
    *,
    title: str | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Trace (left) and per-chain marginal (right) for each parameter.

    Well-mixed chains overlap as stationary noise on the left and give
    matching histograms on the right. ``var_names = None`` plots the sampled
    background parameters, skipping fixed ones.
    """
    if var_names is None:
        var_names = _sampled_parameters(posterior)
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
    var_names: Sequence[str] | None = None,
    *,
    title: str | None = None,
    output_path: Path | None = None,
    corner_kwargs: Mapping[str, Any] | None = None,
) -> Figure:
    """Corner plot of the joint posterior, all chains pooled.

    Titles show the median and the 16th-84th percentile range;
    ``corner_kwargs`` is passed on to ``corner.corner`` and overrides these
    defaults. ``var_names = None`` uses the sampled background parameters,
    skipping fixed ones.
    """
    import corner  # noqa: PLC0415

    if var_names is None:
        var_names = _sampled_parameters(posterior)

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


def plot_prior_predictive(
    stan_data: Mapping[str, Any],
    *,
    n_draws: int = 1000,
    n_lines: int = 30,
    seed: int = 0,
    title: str | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Spectra implied by the priors of ``stan_data``, against the data.

    ``n_draws`` parameter sets are drawn from the lognormal priors
    (``sample_prior_predictive``); fixed parameters keep their values, so
    ``build_stan_data(..., fixed=NO_LINEAR)`` shows the model without the
    linear term. No Stan run is needed.

    Top: data, the 50% and 90% bands and the median of the expected counts,
    ``n_lines`` single prior draws, and the median of each component.
    Middle: the same bands and the data divided by the prior median. Bottom:
    prior-predictive total counts in the fitted bins below and above the
    Ar39 endpoint, with the observed totals and their prior-predictive
    quantile.

    The priors are reasonable when the bands cover the data within a factor
    of a few, the single draws look like plausible spectra, and the observed
    totals sit inside the bulk of the prior-predictive totals. Bands much
    narrower than that make the priors decide the result; bands spanning
    many orders of magnitude or absurd shapes mean the priors allow spectra
    nobody believes.
    """
    prior = sample_prior_predictive(stan_data, n_draws, seed=seed)
    expected = prior.expected_counts
    edges = np.asarray(stan_data["edges"], dtype=float)
    counts = np.asarray(stan_data["counts"], dtype=float)
    used = np.asarray(stan_data["use_bin"], dtype=bool)
    centres = 0.5 * (edges[:-1] + edges[1:])
    q05, q25, q50, q75, q95 = np.percentile(expected, (5, 25, 50, 75, 95), axis=0)

    fig = plt.figure(figsize=(11, 10), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[2.3, 1])
    ax_spec, ax_ratio = top.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1.3]}
    )

    rng = np.random.default_rng(seed)
    for k in rng.choice(len(expected), size=min(n_lines, len(expected)), replace=False):
        ax_spec.stairs(expected[k], edges, color="grey", lw=0.4, alpha=0.4)

    def band(ax: Any, lo: np.ndarray, hi: np.ndarray, alpha: float, **kw: Any) -> None:
        # Repeat the last value so the step band covers the last bin too.
        ax.fill_between(
            edges,
            np.append(lo, lo[-1]),
            np.append(hi, hi[-1]),
            step="post",
            color="C0",
            alpha=alpha,
            lw=0,
            **kw,
        )

    band(ax_spec, q05, q95, 0.18, label="prior 90%")
    band(ax_spec, q25, q75, 0.35, label="prior 50%")
    ax_spec.stairs(q50, edges, color="C0", lw=1.2, label="prior median")
    for name, values in prior.components.items():
        if not np.any(values):
            continue
        color, label = _COMPONENT_STYLE.get(name, (None, name))
        ax_spec.stairs(
            np.median(values, axis=0),
            edges,
            color=color,
            lw=0.9,
            ls="--",
            label=f"{label} (median)",
        )
    ax_spec.stairs(counts, edges, color="black", lw=0.8, label="data")
    positive = q05[q05 > 0]
    bottom_limit = 0.3 * positive.min() if positive.size else 1e-2
    ax_spec.set_ylim(bottom_limit, 3 * max(q95.max(), counts.max()))
    ax_spec.set(ylabel="Counts / bin", yscale="log")
    note = f"{prior.n_rejected} draws rejected" if prior.n_rejected else None
    ax_spec.legend(loc="lower left", fontsize=8, ncols=2, title=note, title_fontsize=7)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(q50 > 0, 1.0 / q50, np.nan)
    band(ax_ratio, q05 * ratio, q95 * ratio, 0.18)
    band(ax_ratio, q25 * ratio, q75 * ratio, 0.35)
    ax_ratio.axhline(1.0, color="C0", lw=1.0)
    ax_ratio.plot(centres[used], (counts * ratio)[used], ".", ms=2, color="black")
    ax_ratio.set(
        yscale="log",
        ylabel="/ prior median",
        xlabel="Energy [keV]",
        xlim=(edges[0], edges[-1]),
    )

    endpoint = float(stan_data["loc"]) + float(stan_data["scale"])
    for idx in _excluded_windows(used):
        for ax in (ax_spec, ax_ratio):
            ax.axvspan(edges[idx[0]], edges[idx[-1] + 1], color="grey", alpha=0.2, lw=0)
    if edges[0] < endpoint < edges[-1]:
        for ax in (ax_spec, ax_ratio):
            ax.axvline(endpoint, color="C0", ls=":", lw=1.0)
        regions = (
            (f"{edges[0]:g}-{endpoint:g} keV", centres < endpoint),
            (f"{endpoint:g}-{edges[-1]:g} keV", centres >= endpoint),
        )
    else:
        regions = ((f"{edges[0]:g}-{edges[-1]:g} keV", np.ones_like(used)),)

    axes = np.atleast_1d(bottom.subplots(1, len(regions)))
    for ax, (label, region) in zip(axes, regions, strict=True):
        mask = region & used
        observed = counts[mask].sum()
        totals = prior.counts_rep[:, mask].sum(axis=1)
        quantile = float(np.mean(totals <= observed))
        logs = np.log10(np.maximum(totals, 1))
        ax.hist(logs, bins=50, color="C0", alpha=0.5)
        ax.axvline(np.log10(max(observed, 1)), color="black", lw=1.2, label="data")
        ax.set(
            xlabel=r"$\log_{10}$ total counts (fitted bins)",
            title=f"{label}: data at prior quantile {quantile:.2f}",
        )
        ax.set_yticks([])
        ax.legend(fontsize=8)
    fig.suptitle(title or "Prior predictive")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig

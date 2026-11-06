# Copyright (C) 2026 Francesco Borra
#

"""Plots that check a comparison between two background-model variants.

Typical use is the full model (``a``) against the model without the linear
continuum (``b``, ``c0 = cQ = 0``) for one detector group. Every function
works from saved fit outputs (``load_background_fit``, ``load_stan_data``,
and ``posterior.nc`` opened with ``xarray.open_datatree``) as well as from
in-memory ``cmdstanpy`` fits, so the checks run where the fits are stored,
without refitting.

The questions they answer:

* :func:`plot_model_difference`: where in energy do the two predictions
  differ, and is that difference small compared with the Poisson noise?
* :func:`plot_pointwise_elpd`: which bins drive the LOO difference, and is
  PSIS reliable there?
* :func:`plot_linear_prior_posterior`: what do the data say about ``c0`` and
  ``cQ`` compared with their priors (the driver of the Bayes factor)?
* :func:`plot_shared_parameters`: does removing the linear term shift the
  other parameters?
* :func:`plot_bf_prior_sensitivity`: does the Bayes factor depend on the
  ``c0``/``cQ`` prior width, and does it agree with LOO?
* :func:`plot_bridge_diagnostics`: is the bridge-sampling estimate stable?
* :func:`plot_ppc_discrepancy`: do both models fit the data at all?
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.stats import norm, skew

from bosonic_dm.fitting.background_fit import BackgroundFitSummary, free_parameters
from bosonic_dm.fitting.model_comparison import BridgeSamplingCheck, pointwise_loo
from bosonic_dm.plotting.background_fit import (
    PARAMETER_LABELS,
    _excluded_windows,
    _sampled_parameters,
    normalized_residuals,
    posterior_draws,
)

AR39_ENDPOINT_KEV = 565.8
_LN10 = np.log(10.0)
# Kass & Raftery (1995) bounds on 2 ln BF (2, 6, 10), expressed in |ln BF|.
_KASS_RAFTERY_LN_BF = (
    (1.0, "bare mention"),
    (3.0, "positive"),
    (5.0, "strong"),
)
_MODEL_COLORS = ("C3", "C0")


def _save(fig: Figure, output_path: Path | None, dpi: int = 200) -> Figure:
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return fig


def _shade_excluded(axes: Sequence[Any], edges: np.ndarray, used: np.ndarray) -> None:
    for idx in _excluded_windows(np.asarray(used, dtype=bool)):
        for ax in axes:
            ax.axvspan(edges[idx[0]], edges[idx[-1] + 1], color="grey", alpha=0.2, lw=0)


def _vector_draws(posterior: Any, name: str, group: str) -> np.ndarray:
    """Draws of a per-bin variable, shaped ``(samples, bins)``, chains pooled."""
    if hasattr(posterior, "draws") and hasattr(posterior, "column_names"):
        return np.asarray(posterior.stan_variable(name), dtype=float)
    values = posterior[group][name]
    values = np.asarray(values.transpose("chain", "draw", ...).values, dtype=float)
    return values.reshape(-1, values.shape[-1])


def plot_model_difference(
    summary_a: BackgroundFitSummary,
    summary_b: BackgroundFitSummary,
    counts: Sequence[int] | np.ndarray,
    *,
    labels: tuple[str, str] = ("full", "no linear"),
    window_keV: float | None = None,
    ar39_endpoint_keV: float | None = AR39_ENDPOINT_KEV,
    residual_limit: float = 5.0,
    title: str | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Spectrum, prediction difference, and pulls of two background variants.

    Top: data, ``B_hat`` of both variants, and the linear component of
    ``a`` (with its total over the fitted bins). Middle:
    ``(B_hat_a - B_hat_b) / sqrt(B_hat_b)`` per bin, the prediction
    difference in units of the Poisson standard deviation; with
    ``window_keV`` also its sum over a sliding window of that width, divided
    by the square root of the windowed ``B_hat_b``, which is what a signal
    peak of that width would see. Bottom: predictive pulls of both variants
    on the fitted bins (see ``normalized_residuals``), with their RMS.

    Above the Ar39 endpoint only the double-beta and linear terms remain, so a
    difference there shows how the two variants share the continuum.
    """
    n = np.asarray(counts, dtype=float)
    edges = summary_a.bin_edges
    if not np.array_equal(edges, summary_b.bin_edges):
        msg = "The two summaries have different bin edges."
        raise ValueError(msg)
    centres = 0.5 * (edges[:-1] + edges[1:])
    used = summary_a.used_in_fit
    b_a, b_b = summary_a.b_hat, summary_b.b_hat
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = np.where(b_b > 0, (b_a - b_b) / np.sqrt(b_b), np.nan)

    fig, (ax_spec, ax_diff, ax_res) = plt.subplots(
        3,
        1,
        figsize=(11, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 1.5], "hspace": 0.06},
    )
    ax_spec.stairs(n, edges, color="black", lw=0.8, label="data")
    ax_spec.stairs(
        b_a, edges, color=_MODEL_COLORS[0], lw=1.3, label=rf"$\hat{{B}}$ {labels[0]}"
    )
    ax_spec.stairs(
        b_b,
        edges,
        color=_MODEL_COLORS[1],
        lw=1.3,
        ls="--",
        label=rf"$\hat{{B}}$ {labels[1]}",
    )
    linear = summary_a.component_medians.get("expected_linear")
    if linear is not None:
        ax_spec.stairs(
            linear,
            edges,
            color="C2",
            lw=1.0,
            label=f"linear, {labels[0]} ({linear[used].sum():.3g} counts fitted)",
        )
    ax_spec.set(ylabel="Counts / bin", yscale="log")
    ax_spec.set_ylim(1e-2, 1.5 * max(np.max(n), np.max(b_a)))
    ax_spec.legend(loc="upper right", fontsize=9)

    ax_diff.axhspan(-1, 1, color="#c9dfa0", lw=0, zorder=0)
    ax_diff.axhline(0.0, color="grey", lw=0.7)
    ax_diff.stairs(diff, edges, color=_MODEL_COLORS[0], lw=0.9, label="per bin")
    if window_keV is not None:
        width = max(1, round(window_keV / float(np.median(np.diff(edges)))))
        kernel = np.ones(width)
        excess = np.convolve(b_a - b_b, kernel, mode="same")
        noise = np.sqrt(np.convolve(b_b, kernel, mode="same"))
        with np.errstate(divide="ignore", invalid="ignore"):
            windowed = np.where(noise > 0, excess / noise, np.nan)
        ax_diff.plot(
            centres, windowed, color="C4", lw=1.2, label=f"{window_keV:g} keV window"
        )
    ax_diff.set_ylabel(r"$\Delta\hat{B} / \sqrt{\hat{B}}$")
    ax_diff.legend(loc="upper right", fontsize=8, ncols=2)

    for k, color in ((2, "#f2e6a0"), (1, "#c9dfa0")):
        ax_res.axhspan(-k, k, color=color, lw=0, zorder=0)
    ax_res.axhline(0.0, color="grey", lw=0.7)
    for summary, label, color in zip(
        (summary_a, summary_b), labels, _MODEL_COLORS, strict=True
    ):
        pulls = normalized_residuals(summary, n)[used]
        rms = float(np.sqrt(np.nanmean(pulls**2)))
        clipped = np.clip(pulls, -0.95 * residual_limit, 0.95 * residual_limit)
        ax_res.plot(
            centres[used],
            clipped,
            ".",
            ms=2.5,
            alpha=0.6,
            color=color,
            label=f"{label}: rms {rms:.2f}",
        )
    ax_res.set_ylim(-residual_limit, residual_limit)
    ax_res.set(xlabel="Energy [keV]", ylabel="pull", xlim=(edges[0], edges[-1]))
    ax_res.legend(loc="upper right", fontsize=8, ncols=2)

    axes = (ax_spec, ax_diff, ax_res)
    _shade_excluded(axes, edges, used)
    if ar39_endpoint_keV is not None and edges[0] < ar39_endpoint_keV < edges[-1]:
        for ax in axes:
            ax.axvline(ar39_endpoint_keV, color="C0", ls=":", lw=1.0)
        ax_spec.annotate(
            r"$^{39}$Ar endpoint",
            xy=(ar39_endpoint_keV, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -4),
            textcoords="offset points",
            va="top",
            fontsize=8,
            color="C0",
        )
    ax_spec.set_title(title or f"{labels[0]} vs {labels[1]}")
    return _save(fig, output_path)


def plot_pointwise_elpd(
    posterior_a: Any,
    posterior_b: Any,
    stan_data: Mapping[str, Any],
    *,
    labels: tuple[str, str] = ("full", "no linear"),
    k_threshold: float = 0.7,
    title: str | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Per-bin PSIS-LOO difference of two variants and their Pareto ``k``.

    Top: ``elpd_i(a) - elpd_i(b)`` for each fitted bin, positive where ``a``
    predicts the left-out bin better; the legend gives the sum and its
    standard error. Middle: the cumulative sum along energy, which shows
    whether the total comes from the whole range or from a few bins. Bottom:
    Pareto ``k`` of both variants; above ``k_threshold`` the PSIS estimate
    of that bin is unreliable. ``stan_data`` is the fitted data of either
    variant (they share bins and counts).
    """
    elpd_a, k_a = pointwise_loo(posterior_a, stan_data)
    elpd_b, k_b = pointwise_loo(posterior_b, stan_data)
    edges = np.asarray(stan_data["edges"], dtype=float)
    used = np.asarray(stan_data["use_bin"], dtype=bool)
    centres = 0.5 * (edges[:-1] + edges[1:])[used]
    delta = elpd_a - elpd_b
    total = float(delta.sum())
    se = float(np.sqrt(len(delta) * np.var(delta)))

    fig, (ax_d, ax_cum, ax_k) = plt.subplots(
        3,
        1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.2, 1.2], "hspace": 0.08},
    )
    ax_d.axhline(0.0, color="grey", lw=0.7)
    ax_d.plot(
        centres,
        delta,
        ".",
        ms=2.5,
        color="black",
        label=rf"$\Sigma$ = {total:+.2f} $\pm$ {se:.2f}",
    )
    ax_d.set_ylabel(r"$\Delta\,\mathrm{elpd}_i$")
    ax_d.legend(loc="upper right", fontsize=9, title=f"{labels[0]} - {labels[1]}")

    ax_cum.axhline(0.0, color="grey", lw=0.7)
    ax_cum.plot(centres, np.cumsum(delta), color=_MODEL_COLORS[0], lw=1.2)
    ax_cum.set_ylabel(r"cumulative $\Delta\,\mathrm{elpd}$")

    for k, label, color in zip((k_a, k_b), labels, _MODEL_COLORS, strict=True):
        n_bad = int(np.sum(k > k_threshold))
        ax_k.plot(
            centres,
            k,
            ".",
            ms=2.5,
            alpha=0.7,
            color=color,
            label=f"{label}: max {np.nanmax(k):.2f}, {n_bad} above {k_threshold:g}",
        )
    ax_k.axhline(k_threshold, color="grey", ls="--", lw=0.8)
    ax_k.set(
        xlabel="Energy [keV]", ylabel=r"Pareto $\hat{k}$", xlim=(edges[0], edges[-1])
    )
    ax_k.legend(loc="upper right", fontsize=8)

    _shade_excluded((ax_d, ax_cum, ax_k), edges, used)
    ax_d.set_title(title or "PSIS-LOO, leave one bin out")
    return _save(fig, output_path)


def plot_linear_prior_posterior(
    posterior: Any,
    stan_data: Mapping[str, Any],
    *,
    prior_sds: Sequence[float] = (0.5, 1.0, 2.0),
    sensitivity_posteriors: Mapping[float, Any] | None = None,
    n_prior_draws: int = 20000,
    seed: int = 0,
    title: str | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Prior and posterior of the linear-continuum coefficients.

    Panels: ``log10 c0`` and ``log10 cQ`` (posterior histogram, lognormal
    priors for each width in ``prior_sds``); their joint posterior with the
    1-sigma and 2-sigma ellipses of the fitted prior; and the linear counts in the
    fitted bins as a fraction of the observed fitted counts, posterior
    against the fitted prior.

    A posterior that matches the prior means the data say little about the
    linear term; one pushed to the low tail means the data disfavour it. The
    Bayes factor and its trend with the prior width follow from this.
    ``sensitivity_posteriors`` maps further prior widths to fits made with
    them (e.g. the prior-sensitivity refits); they are drawn in the colour of
    their prior.
    """
    if not {"c0", "cQ"} <= set(free_parameters(stan_data)):
        msg = "c0 and cQ must both be free in stan_data."
        raise ValueError(msg)
    names = ("c0", "cQ")
    draws = posterior_draws(posterior, names).reshape(-1, 2)
    means = np.asarray(stan_data["log_rate_prior_mean"], dtype=float)[2:4]
    fit_sds = np.asarray(stan_data["log_rate_prior_sd"], dtype=float)[2:4]
    prior_sds = tuple(float(sd) for sd in prior_sds)
    colors = plt.cm.Blues(np.linspace(0.45, 0.95, len(prior_sds)))
    color_of = dict(zip(prior_sds, colors, strict=True))
    extra = {
        float(sd): posterior_draws(p, names).reshape(-1, 2)
        for sd, p in (sensitivity_posteriors or {}).items()
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout="constrained")
    widest = max(*prior_sds, *fit_sds)
    for k, (name, ax) in enumerate(zip(names, axes[:2], strict=True)):
        x = np.log10(draws[:, k])
        lo = min(x.min(), (means[k] - 3 * widest) / _LN10)
        hi = max(x.max(), (means[k] + 3 * widest) / _LN10)
        grid = np.linspace(lo, hi, 400)
        bins = np.linspace(x.min(), x.max(), 50)
        ax.hist(
            x,
            bins=bins,
            density=True,
            color=_MODEL_COLORS[0],
            alpha=0.35,
            label="posterior",
        )
        for sd in prior_sds:
            fitted = np.isclose(sd, fit_sds[k])
            ax.plot(
                grid,
                norm.pdf(grid, means[k] / _LN10, sd / _LN10),
                color=color_of[sd],
                lw=1.8 if fitted else 1.0,
                label=f"prior sd {sd:g}" + (" (this fit)" if fitted else ""),
            )
        for sd, values in extra.items():
            ax.hist(
                np.log10(values[:, k]),
                bins=50,
                density=True,
                histtype="step",
                color=color_of.get(sd, "grey"),
                label=f"posterior, prior sd {sd:g}",
            )
        label = PARAMETER_LABELS.get(name, name)
        ax.set(xlabel=rf"$\log_{{10}}$ {label} [counts/keV]", ylabel="density")
    axes[0].legend(fontsize=7)

    ax = axes[2]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(draws), size=min(4000, len(draws)), replace=False)
    log_draws = np.log10(draws[pick])
    ax.scatter(log_draws[:, 0], log_draws[:, 1], s=2, alpha=0.3, color=_MODEL_COLORS[0])
    centre = means / _LN10
    for n_sigma in (1, 2):
        ax.add_patch(
            Ellipse(
                centre,
                2 * n_sigma * fit_sds[0] / _LN10,
                2 * n_sigma * fit_sds[1] / _LN10,
                fill=False,
                color=color_of.get(float(fit_sds[0]), "C0"),
                ls="-" if n_sigma == 1 else "--",
            )
        )
    ax.plot(*centre, "+", color="C0", ms=10, label="prior centre")
    ax.set(
        xlabel=rf"$\log_{{10}}$ {PARAMETER_LABELS['c0']}",
        ylabel=rf"$\log_{{10}}$ {PARAMETER_LABELS['cQ']}",
    )
    ax.autoscale_view()
    ax.legend(fontsize=7, title=r"prior $1\sigma$, $2\sigma$", title_fontsize=7)

    ax = axes[3]
    edges = np.asarray(stan_data["edges"], dtype=float)
    used = np.asarray(stan_data["use_bin"], dtype=bool)
    q_value = float(stan_data["Q"])
    width = np.diff(edges)
    midpoint_fraction = (edges[1:] + edges[:-1]) / (2 * q_value)
    integrals = np.array(
        [
            (width * (1 - midpoint_fraction))[used].sum(),
            (width * midpoint_fraction)[used].sum(),
        ]
    )
    observed = float(np.asarray(stan_data["counts"], dtype=float)[used].sum())
    posterior_fraction = draws @ integrals / observed
    prior_draws = np.exp(rng.normal(means, fit_sds, size=(n_prior_draws, 2)))
    prior_fraction = prior_draws @ integrals / observed
    both = np.log10(np.r_[posterior_fraction, prior_fraction])
    bins = np.linspace(np.percentile(both, 0.1), np.percentile(both, 99.9), 60)
    ax.hist(
        np.log10(posterior_fraction),
        bins=bins,
        density=True,
        color=_MODEL_COLORS[0],
        alpha=0.35,
        label="posterior",
    )
    ax.hist(
        np.log10(prior_fraction),
        bins=bins,
        density=True,
        histtype="step",
        color=color_of.get(float(fit_sds[0]), "C0"),
        label="prior (this fit)",
    )
    median = float(np.median(posterior_fraction))
    ax.set(
        xlabel=r"$\log_{10}$ (linear counts / fitted counts)",
        ylabel="density",
        title=f"posterior median {median:.3g}",
    )
    ax.legend(fontsize=7)
    if title:
        fig.suptitle(title)
    return _save(fig, output_path)


def plot_shared_parameters(
    posterior_a: Any,
    posterior_b: Any,
    *,
    labels: tuple[str, str] = ("full", "no linear"),
    title: str | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Posterior intervals of the parameters free in both variants.

    Each row shows the median, the 68% and the 95% equal-tailed interval of
    one parameter in both fits, and on the right the shift of the medians in
    units of the combined 68% half-widths. A large shift of ``N_2nu`` or
    ``N_ar39`` means the other components absorb what the linear term
    described.
    """
    names_b = set(_sampled_parameters(posterior_b))
    shared = [name for name in _sampled_parameters(posterior_a) if name in names_b]
    draws = [
        posterior_draws(p, shared).reshape(-1, len(shared))
        for p in (posterior_a, posterior_b)
    ]
    fig, axes = plt.subplots(
        len(shared),
        1,
        figsize=(8, 0.75 * len(shared) + 1.2),
        squeeze=False,
        layout="constrained",
    )
    for k, name in enumerate(shared):
        ax = axes[k, 0]
        medians, half_widths = [], []
        for row, (d, color) in enumerate(zip(draws, _MODEL_COLORS, strict=True)):
            lo95, lo68, med, hi68, hi95 = np.percentile(
                d[:, k], (2.5, 16, 50, 84, 97.5)
            )
            y = 1 - row
            ax.hlines(y, lo95, hi95, color=color, lw=1.0)
            ax.hlines(y, lo68, hi68, color=color, lw=3.5)
            ax.plot(med, y, "o", color="white", mec=color, ms=5)
            medians.append(med)
            half_widths.append(0.5 * (hi68 - lo68))
        shift = (medians[0] - medians[1]) / np.hypot(*half_widths)
        ax.set_ylim(-0.8, 1.8)
        ax.set_yticks([])
        ax.set_ylabel(
            PARAMETER_LABELS.get(name, name), rotation=0, ha="right", va="center"
        )
        ax.text(
            1.01,
            0.5,
            rf"{shift:+.2f}$\sigma$",
            transform=ax.transAxes,
            va="center",
            fontsize=9,
        )
        ax.tick_params(axis="x", labelsize=8)
    handles = [Line2D([], [], color=c, lw=3.5) for c in _MODEL_COLORS]
    axes[0, 0].legend(handles, labels, loc="lower right", fontsize=8, ncols=2)
    fig.suptitle(title or "Shared parameters: median, 68% and 95% intervals")
    return _save(fig, output_path)


def _signed_loo(record: Mapping[str, Any]) -> tuple[float, float]:
    """``elpd(a) - elpd(b)`` and its standard error from a comparison record."""
    model_a = record["bayes_factor"]["model_a"]
    model_b = record["bayes_factor"]["model_b"]
    rows = {row["model"]: row for row in record["loo"]}
    if rows[model_a]["rank"] == 0:
        worse, sign = rows[model_b], 1.0
    else:
        worse, sign = rows[model_a], -1.0
    dse = worse.get("dse")
    return sign * abs(float(worse["elpd_diff"])), float(
        "nan"
    ) if dse is None else float(dse)


def plot_bf_prior_sensitivity(
    records: Mapping[str, Mapping[str, Any]],
    *,
    title: str | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Bayes factor against the ``c0``/``cQ`` prior width, with LOO alongside.

    ``records`` maps a group name to its parsed ``model_comparison.yaml``.
    Left: ``ln BF(a/b)`` with its bridge-sampling error for each prior width,
    one line per group, on the Kass & Raftery bands. Right: the PSIS-LOO
    difference ``elpd(a) - elpd(b)`` with its standard error, which depends
    only weakly on the priors. Where both are compatible with zero the data
    cannot tell the variants apart.
    """
    fig, (ax_bf, ax_loo) = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        gridspec_kw={"width_ratios": [1.6, 1]},
        layout="constrained",
    )
    for bound, label in _KASS_RAFTERY_LN_BF:
        ax_bf.axhspan(-bound, bound, color="grey", alpha=0.1, lw=0)
        ax_bf.annotate(
            label,
            xy=(1.0, bound),
            xycoords=("axes fraction", "data"),
            xytext=(-3, -2),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=7,
            color="grey",
        )
    ax_bf.axhline(0.0, color="grey", lw=0.7)
    n_groups = len(records)
    widths = set()
    model_a = model_b = None
    for i, (group, record) in enumerate(records.items()):
        rows = sorted(record["prior_sensitivity"], key=lambda r: r["c0_cQ_prior_sd"])
        x = np.array([r["c0_cQ_prior_sd"] for r in rows], dtype=float)
        widths.update(x.tolist())
        nudge = 10 ** (0.012 * (i - (n_groups - 1) / 2))
        ax_bf.errorbar(
            x * nudge,
            [r["log_bf"] for r in rows],
            yerr=[r["log_bf_sd"] for r in rows],
            marker="o",
            ms=4,
            capsize=3,
            color=f"C{i}",
            label=group,
        )
        model_a = record["bayes_factor"]["model_a"]
        model_b = record["bayes_factor"]["model_b"]
        delta, dse = _signed_loo(record)
        ax_loo.errorbar(delta, i, xerr=dse, fmt="o", capsize=3, color=f"C{i}")
    ax_bf.set_xscale("log")
    ticks = sorted(widths)
    ax_bf.set_xticks(ticks, labels=[f"{w:g}" for w in ticks])
    ax_bf.minorticks_off()
    ax_bf.set(
        xlabel=r"$c_0$, $c_Q$ prior sd (log space)",
        ylabel=f"ln BF({model_a} / {model_b})",
    )
    ax_bf.legend(fontsize=8)
    ax_loo.axvline(0.0, color="grey", lw=0.7)
    ax_loo.set_yticks(range(n_groups), labels=list(records))
    ax_loo.set_ylim(n_groups - 0.5, -0.5)
    ax_loo.set_xlabel(f"elpd({model_a}) - elpd({model_b})")
    ax_loo.set_title("PSIS-LOO", fontsize=10)
    if title:
        fig.suptitle(title)
    return _save(fig, output_path)


def plot_bridge_diagnostics(
    check: BridgeSamplingCheck,
    *,
    title: str | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Stability and proposal checks of one bridge-sampling estimate.

    ``check`` is the output of ``check_bridge_sampling``. Top row:
    ``ln q - ln g`` at the posterior and at the proposal draws (the two
    histograms must overlap, or the estimate rests on few draws); ``ln Z``
    for other proposal seeds, whose spread should match the quoted standard
    deviation (grey band); ``ln Z`` against the draws per chain used, which
    should settle. Bottom: each free parameter on the log scale, posterior
    against the normal proposal; strong skewness or heavy tails make the
    proposal a poor bridge.
    """
    terms = check.terms
    ref = check.result
    l_post = terms.log_q_posterior - terms.log_g_posterior
    l_prop = terms.log_q_proposal - terms.log_g_proposal
    n_zero = int(np.sum(~np.isfinite(l_prop)))
    l_prop = l_prop[np.isfinite(l_prop)]

    fig = plt.figure(figsize=(14, 8), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.1])
    ax_w, ax_seed, ax_n = top.subplots(1, 3)

    both = np.r_[l_post, l_prop]
    bins = np.linspace(np.percentile(both, 0.5), np.percentile(both, 99.9), 80)
    ax_w.hist(
        l_post,
        bins,
        density=True,
        histtype="step",
        color=_MODEL_COLORS[0],
        label="posterior draws",
    )
    ax_w.hist(
        l_prop,
        bins,
        density=True,
        histtype="step",
        color=_MODEL_COLORS[1],
        label="proposal draws",
    )
    ax_w.axvline(ref.log_ml, color="black", ls="--", lw=0.8, label=r"$\ln Z$")
    ax_w.set(xlabel=r"$\ln q - \ln g$", ylabel="density")
    note = f"{n_zero} proposal draws with q = 0" if n_zero else None
    ax_w.legend(fontsize=8, title=note, title_fontsize=7)

    for ax in (ax_seed, ax_n):
        ax.axhspan(
            ref.log_ml - ref.log_ml_sd,
            ref.log_ml + ref.log_ml_sd,
            color="grey",
            alpha=0.25,
            lw=0,
        )
        ax.axhline(ref.log_ml, color="grey", lw=0.8)
        ax.ticklabel_format(axis="y", useOffset=ref.log_ml)
    values = np.array([r.log_ml for r in check.by_seed])
    ax_seed.errorbar(
        check.seeds,
        values,
        [r.log_ml_sd for r in check.by_seed],
        fmt="o",
        ms=3,
        capsize=2,
        color=_MODEL_COLORS[0],
    )
    spread = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    ax_seed.set(
        xlabel="proposal seed",
        ylabel=r"$\ln Z$",
        title=f"seed spread {spread:.3f}, quoted sd {ref.log_ml_sd:.3f}",
    )
    ax_n.errorbar(
        check.n_draws,
        [r.log_ml for r in check.by_draws],
        [r.log_ml_sd for r in check.by_draws],
        fmt="o-",
        ms=3,
        capsize=2,
        color=_MODEL_COLORS[1],
    )
    ax_n.set_xscale("log", base=2)
    ax_n.set_xticks(check.n_draws, labels=[str(n) for n in check.n_draws])
    ax_n.minorticks_off()
    ax_n.set(
        xlabel="draws per chain used", ylabel=r"$\ln Z$", title="convergence with draws"
    )

    dim = len(check.parameters)
    ncols = min(dim, 5)
    nrows = int(np.ceil(dim / ncols))
    axes = bottom.subplots(nrows, ncols, squeeze=False)
    draws = terms.posterior_draws.reshape(-1, dim)
    sds = np.sqrt(np.diag(terms.proposal_cov))
    for k, name in enumerate(check.parameters):
        ax = axes.flat[k]
        x = draws[:, k]
        mean, sd = terms.proposal_mean[k], sds[k]
        grid = np.linspace(
            min(x.min(), mean - 4 * sd), max(x.max(), mean + 4 * sd), 300
        )
        ax.hist(
            x,
            bins=50,
            density=True,
            color=_MODEL_COLORS[0],
            alpha=0.35,
            label="posterior",
        )
        ax.plot(
            grid, norm.pdf(grid, mean, sd), color=_MODEL_COLORS[1], label="proposal"
        )
        ax.text(
            0.03,
            0.95,
            f"skew {skew(x):+.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        ax.set_xlabel(f"ln {PARAMETER_LABELS.get(name, name)}")
        ax.set_yticks([])
    for ax in axes.flat[dim:]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7, loc="upper right")
    bottom.suptitle("Posterior vs normal proposal (log scale)", fontsize=10)
    fig.suptitle(
        title or f"Bridge sampling: ln Z = {ref.log_ml:.3f} ± {ref.log_ml_sd:.3f}"
    )
    return _save(fig, output_path)


def plot_ppc_discrepancy(
    posterior_a: Any,
    posterior_b: Any,
    stan_data: Mapping[str, Any],
    *,
    labels: tuple[str, str] = ("full", "no linear"),
    slice_keV: float = 100.0,
    max_draws: int = 2000,
    seed: int = 0,
    title: str | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Posterior-predictive checks of both variants on the fitted bins.

    First two panels: the Pearson discrepancy
    ``T(y, θ) = Σ (y - λ)² / λ`` of the observed counts against that of
    replicated counts drawn with the same ``θ``; the posterior-predictive
    p-value ``P(T(y_rep) ≥ T(y))`` is in the title, and values near 0 or 1
    flag a misfit. Last panel: observed minus predicted counts summed in
    ``slice_keV`` slices, in units of the predictive standard deviation, to
    locate a misfit in energy. At most ``max_draws`` random posterior draws
    are used.
    """
    counts = np.asarray(stan_data["counts"], dtype=float)
    used = np.asarray(stan_data["use_bin"], dtype=bool)
    edges = np.asarray(stan_data["edges"], dtype=float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    slice_id = np.floor((centres - edges[0]) / slice_keV).astype(int)
    n_slices = int(slice_id.max()) + 1
    indicator = np.zeros((int(used.sum()), n_slices))
    indicator[np.arange(indicator.shape[0]), slice_id[used]] = 1.0
    slice_centres = edges[0] + (np.arange(n_slices) + 0.5) * slice_keV
    y = counts[used]
    observed_slices = y @ indicator

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), layout="constrained")
    rng = np.random.default_rng(seed)
    for j, (posterior, label, color) in enumerate(
        zip((posterior_a, posterior_b), labels, _MODEL_COLORS, strict=True)
    ):
        expected = _vector_draws(posterior, "expected_counts", "posterior")
        replicated = _vector_draws(posterior, "counts_rep", "posterior_predictive")
        pick = rng.choice(
            len(expected), size=min(max_draws, len(expected)), replace=False
        )
        lam = expected[pick][:, used]
        rep = replicated[pick][:, used]
        t_obs = ((y - lam) ** 2 / lam).sum(axis=1)
        t_rep = ((rep - lam) ** 2 / lam).sum(axis=1)
        p_value = float(np.mean(t_rep >= t_obs))
        ax = axes[j]
        ax.scatter(t_obs, t_rep, s=3, alpha=0.3, color=color)
        lo = min(t_obs.min(), t_rep.min())
        hi = max(t_obs.max(), t_rep.max())
        ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8)
        ax.set(
            xlabel=r"$T(y, \theta)$",
            ylabel=r"$T(y^{\mathrm{rep}}, \theta)$",
            title=f"{label}: p = {p_value:.2f} ({int(used.sum())} bins)",
        )
        predicted = rep @ indicator
        spread = predicted.std(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(
                spread > 0, (observed_slices - predicted.mean(axis=0)) / spread, np.nan
            )
        offset = (j - 0.5) * 0.08 * slice_keV
        axes[2].plot(slice_centres + offset, z, "o", ms=4, color=color, label=label)
    ax = axes[2]
    for k, shade in ((2, "#f2e6a0"), (1, "#c9dfa0")):
        ax.axhspan(-k, k, color=shade, lw=0, zorder=0)
    ax.axhline(0.0, color="grey", lw=0.7)
    ax.set(
        xlabel="Energy [keV]",
        ylabel=r"(obs $-$ pred) / sd$_\mathrm{pred}$",
        title=f"{slice_keV:g} keV slices, fitted bins",
    )
    ax.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    return _save(fig, output_path)

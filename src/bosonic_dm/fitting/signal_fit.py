# Copyright (C) 2026 Francesco Borra
#

"""Fixed-background signal fit at one DM mass and its credible upper limit.

The background ``B_hat`` and the response inputs (efficiency, exposure,
resolution) are fixed at their nominal values. The reported limit is therefore
the one-sided credible upper limit on the signal rate *conditional on the fixed
background and the nominal response*; background and response uncertainties are
not propagated.

Two engines compute the same posterior. ``rate_posterior_quantiles`` integrates
the one-parameter posterior on a grid of rates and handles many
pseudo-experiments at once; it is the engine of the sensitivity study. The Stan
model (``run_signal_fit``) is kept as a cross-check and for later joint fits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from bosonic_dm.fitting.diagnostics import diagnostics_passed, sampler_diagnostics
from bosonic_dm.fitting.signal_template import SignalWindow
from bosonic_dm.fitting.stan_utils import STAN_DIR, sampler_record, software_record
from bosonic_dm.yaml_io import write_yaml

if TYPE_CHECKING:
    import cmdstanpy

SIGNAL_STAN_FILE = STAN_DIR / "signal.stan"

CONDITIONAL_LABEL = (
    "credible upper limit conditional on the fixed background B_hat "
    "and the nominal signal response"
)


@dataclass(frozen=True)
class SignalFitSummary:
    """Posterior summary of the signal rate and sampler diagnostics."""

    credibility: float
    rate: dict[str, float]
    diagnostics: dict[str, Any]
    passed: bool
    rate_draws: np.ndarray

    @property
    def upper_limit(self) -> float:
        """One-sided credible upper limit on the rate, in counts/(kg yr)."""
        return self.rate["upper_limit"]


def suggest_rate_upper(window: SignalWindow, n_sigma: float = 10.0) -> float:
    """Return an upper bound for the uniform rate prior.

    The bound is the rate at which the signal alone predicts all observed
    counts in the fitted bins plus ``n_sigma`` Poisson standard deviations,
    so the likelihood is negligible above it. Check that the posterior stays
    well below the bound (``summarize_signal_fit`` records the ratio).
    """
    use_bin = window.stacked("use_bin").astype(bool)
    observed = float(window.stacked("counts")[use_bin].sum())
    template = float(window.stacked("signal_template")[use_bin].sum())
    if template <= 0:
        msg = "The signal template is zero in every fitted bin."
        raise ValueError(msg)
    return (observed + n_sigma * np.sqrt(observed) + n_sigma) / template


def build_signal_stan_data(window: SignalWindow, rate_upper: float) -> dict[str, Any]:
    """Convert a ``SignalWindow`` into the ``signal.stan`` data block."""
    if not np.isfinite(rate_upper) or rate_upper <= 0:
        msg = f"rate_upper must be finite and positive; got {rate_upper}."
        raise ValueError(msg)
    edges = np.asarray(window.bin_edges, dtype=float)
    counts = window.stacked("counts").astype(int)
    if np.any(counts < 0):
        msg = "Counts must be nonnegative."
        raise ValueError(msg)
    return {
        "n_groups": len(window.groups),
        "n_bins": len(edges) - 1,
        "edges": edges,
        "counts": counts,
        "use_bin": window.stacked("use_bin").astype(int),
        "b_hat": window.stacked("b_hat").astype(float),
        "signal_template": window.stacked("signal_template").astype(float),
        "rate_upper": float(rate_upper),
    }


def run_signal_fit(
    model: cmdstanpy.CmdStanModel,
    stan_data: Mapping[str, Any],
    *,
    seed: int,
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 2000,
    adapt_delta: float = 0.9,
    output_dir: str | Path | None = None,
    show_progress: bool = False,
) -> cmdstanpy.CmdStanMCMC:
    """Sample the signal-rate posterior at one mass."""
    return model.sample(
        data=dict(stan_data),
        chains=chains,
        parallel_chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        adapt_delta=adapt_delta,
        seed=seed,
        output_dir=None if output_dir is None else str(output_dir),
        show_progress=show_progress,
    )


def summarize_signal_fit(
    fit: cmdstanpy.CmdStanMCMC,
    stan_data: Mapping[str, Any],
    *,
    credibility: float = 0.9,
    r_hat_max: float = 1.01,
    ess_min: float = 400.0,
    e_bfmi_min: float = 0.3,
    max_limit_to_bound: float = 0.5,
) -> SignalFitSummary:
    """Summarize the rate posterior and compute the credible upper limit.

    The upper limit is the ``credibility`` quantile of the rate draws. The
    fit fails the checks if the limit exceeds ``max_limit_to_bound`` times the
    prior upper bound, since the truncated prior would then affect the limit.
    """
    import arviz as az  # noqa: PLC0415

    if not 0 < credibility < 1:
        msg = "credibility must lie in (0, 1)."
        raise ValueError(msg)
    idata = az.from_cmdstanpy(posterior=fit)
    table = az.summary(idata, var_names=["R"], round_to="none")
    mcse_limit = az.mcse(idata, var_names=["R"], method="quantile", prob=credibility)

    draws = fit.stan_variable("R")
    upper_limit = float(np.quantile(draws, credibility))
    rate_upper = float(stan_data["rate_upper"])
    rate = {
        "mean": float(draws.mean()),
        "sd": float(draws.std(ddof=1)),
        "median": float(np.median(draws)),
        "upper_limit": upper_limit,
        "upper_limit_mcse": float(mcse_limit["R"].values),
        "r_hat": float(table.loc["R", "r_hat"]),
        "ess_bulk": float(table.loc["R", "ess_bulk"]),
        "ess_tail": float(table.loc["R", "ess_tail"]),
    }
    diagnostics = sampler_diagnostics(fit, {"R": rate})
    diagnostics["upper_limit_to_prior_bound"] = upper_limit / rate_upper
    passed = (
        diagnostics_passed(
            diagnostics, r_hat_max=r_hat_max, ess_min=ess_min, e_bfmi_min=e_bfmi_min
        )
        and diagnostics["upper_limit_to_prior_bound"] <= max_limit_to_bound
    )
    return SignalFitSummary(
        credibility=float(credibility),
        rate=rate,
        diagnostics=diagnostics,
        passed=bool(passed),
        rate_draws=np.asarray(draws, dtype=float),
    )


def save_signal_fit(
    fit: cmdstanpy.CmdStanMCMC,
    summary: SignalFitSummary,
    window: SignalWindow,
    stan_data: Mapping[str, Any],
    output_dir: str | Path,
    *,
    provenance: Mapping[str, Any] | None = None,
    stan_file: str | Path = SIGNAL_STAN_FILE,
) -> dict[str, Path]:
    """Write the rate draws and a YAML record of one mass fit.

    Files written in ``output_dir``: ``rate_draws.npy`` and ``summary.yaml``.
    The YAML records the window, the per-group response inputs, the prior,
    the rate summary and limit, sampler settings and diagnostics, software
    versions, and the SHA-256 of the background-fit files used for ``B_hat``.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {"draws": output / "rate_draws.npy", "summary": output / "summary.yaml"}
    np.save(paths["draws"], summary.rate_draws)

    groups = {}
    for name, inputs in window.groups.items():
        used = inputs.use_bin.astype(bool)
        groups[name] = {
            "exposure_kg_yr": inputs.exposure_kg_yr,
            "efficiency": inputs.efficiency,
            "efficiency_unc": inputs.efficiency_unc,
            "efficiency_exposure_kg_yr": inputs.efficiency_exposure_kg_yr,
            "centroid_keV": inputs.centroid_keV,
            "centroid_offset_keV": inputs.centroid_offset_keV,
            "centroid_offset_unc_keV": inputs.centroid_offset_unc_keV,
            "centroid_offset_extrapolated": inputs.centroid_offset_extrapolated,
            "fwhm_keV": inputs.fwhm_keV,
            "fwhm_unc_keV": inputs.fwhm_unc_keV,
            "sigma_keV": inputs.sigma_keV,
            "window_containment": inputs.window_containment,
            "n_fit_bins": int(used.sum()),
            "observed_counts_fit_bins": int(inputs.counts[used].sum()),
            "b_hat_fit_bins": float(inputs.b_hat[used].sum()),
            "signal_exposure_fit_bins_kg_yr": float(inputs.signal_template[used].sum()),
        }

    edges = window.bin_edges
    write_yaml(
        {
            "mass_keV": window.mass_keV,
            "interpretation": CONDITIONAL_LABEL,
            "rate_units": "counts / (kg yr)",
            "credibility": summary.credibility,
            "rate": summary.rate,
            "passed_diagnostics": summary.passed,
            "diagnostics": summary.diagnostics,
            "window": {
                "half_width_keV": window.half_width_keV,
                "range_keV": [float(edges[0]), float(edges[-1])],
                "n_bins": len(edges) - 1,
                "bin_edges_keV": edges,
            },
            "signal_model": "single Gaussian per group, centroid = mass + fixed "
            "exposure-weighted calibration offset (zero if no residuals given), "
            "FWHM = exposure-weighted mean of detector/run sqrt(a + b E)",
            "groups": groups,
            "prior": {"rate": "uniform", "rate_upper": float(stan_data["rate_upper"])},
            "sampler": sampler_record(fit),
            "software": software_record(stan_file),
            "background_fit_sha256": window.background_sha256,
            "provenance": dict(provenance or {}),
            "files": {name: path.name for name, path in paths.items()},
        },
        paths["summary"],
    )
    return paths


# The grid reaches rates where the likelihood is below 1e-8 of its maximum; the
# posterior beyond is neglected, so any prior bound above the grid end gives
# the same result.
GRID_LOG_DROP = float(np.log(1e8))


def _log_likelihood(
    counts: np.ndarray,
    b_hat: np.ndarray,
    signal_template: np.ndarray,
    template_total: float,
    rates: np.ndarray,
) -> np.ndarray:
    """Poisson log-likelihood up to an R-independent constant.

    ``counts`` has shape ``(datasets, bins)``, ``rates`` ``(datasets, grid)``;
    the result has the shape of ``rates``.
    """
    expected = b_hat + rates[:, :, None] * signal_template
    return np.einsum("kb,kgb->kg", counts, np.log(expected)) - rates * template_total


def _grid_quantiles(
    rates: np.ndarray, log_like: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    """Posterior quantiles from a log-likelihood on per-dataset rate grids."""
    density = np.exp(log_like - log_like.max(axis=1, keepdims=True))
    areas = 0.5 * (density[:, 1:] + density[:, :-1]) * np.diff(rates, axis=1)
    cdf = np.concatenate([np.zeros((len(rates), 1)), np.cumsum(areas, axis=1)], axis=1)
    cdf /= cdf[:, -1:]
    rows = np.arange(len(rates))
    output = np.empty((len(rates), len(quantiles)))
    for column, probability in enumerate(quantiles):
        upper = np.clip((cdf < probability).sum(axis=1), 1, rates.shape[1] - 1)
        c0, c1 = cdf[rows, upper - 1], cdf[rows, upper]
        r0, r1 = rates[rows, upper - 1], rates[rows, upper]
        output[:, column] = r0 + (probability - c0) / (c1 - c0) * (r1 - r0)
    return output


def _chunk_quantiles(
    counts: np.ndarray,
    b_hat: np.ndarray,
    signal_template: np.ndarray,
    template_total: float,
    rate_high: np.ndarray,
    quantiles: np.ndarray,
    n_grid: int,
    n_coarse: int,
    max_doublings: int,
) -> np.ndarray:
    """Two-pass grid posterior for one chunk of datasets.

    The coarse pass on ``[0, rate_high]`` doubles ``rate_high`` until the
    log-likelihood at the end lies ``GRID_LOG_DROP`` below the grid maximum.
    The log-likelihood is concave in R, so the rates above that threshold form
    one interval, bracketed by the neighbouring coarse points. The fine pass
    integrates the posterior on ``n_grid`` points inside that bracket.
    """
    unit = np.linspace(0.0, 1.0, n_coarse)
    high = rate_high.copy()
    pending = np.arange(len(counts))
    low_edge = np.zeros(len(counts))
    high_edge = np.empty(len(counts))
    for _ in range(max_doublings + 1):
        rates = high[pending, None] * unit
        log_like = _log_likelihood(
            counts[pending], b_hat, signal_template, template_total, rates
        )
        above = log_like >= log_like.max(axis=1, keepdims=True) - GRID_LOG_DROP
        resolved = ~above[:, -1]
        rows = np.flatnonzero(resolved)
        first = above[rows].argmax(axis=1)
        last = n_coarse - 1 - above[rows, ::-1].argmax(axis=1)
        low_edge[pending[rows]] = rates[rows, np.maximum(first - 1, 0)]
        high_edge[pending[rows]] = rates[rows, np.minimum(last + 1, n_coarse - 1)]
        pending = pending[~resolved]
        if pending.size == 0:
            break
        high[pending] *= 2.0
    else:
        msg = (
            f"The rate posterior of {pending.size} dataset(s) did not fall by "
            f"exp({GRID_LOG_DROP:.1f}) after {max_doublings} doublings of the grid."
        )
        raise RuntimeError(msg)

    rates = low_edge[:, None] + (high_edge - low_edge)[:, None] * np.linspace(
        0.0, 1.0, n_grid
    )
    log_like = _log_likelihood(counts, b_hat, signal_template, template_total, rates)
    return _grid_quantiles(rates, log_like, quantiles)


def rate_posterior_quantiles(
    counts: np.ndarray,
    b_hat: np.ndarray,
    signal_template: np.ndarray,
    use_bin: np.ndarray,
    quantiles: Sequence[float] = (0.9,),
    *,
    n_grid: int = 1000,
    n_coarse: int = 200,
    n_sigma: float = 10.0,
    template_rtol: float = 1e-12,
    max_doublings: int = 60,
    memory_bytes: float = 2e8,
) -> np.ndarray:
    """Quantiles of the fixed-background rate posterior, by grid integration.

    For every dataset ``k`` the posterior of the rate ``R >= 0`` under a
    uniform prior is proportional to

        prod_(g,i used) Poisson(counts[k, g, i] | b_hat[g, i] + R * s[g, i]),

    the model of ``stan/signal.stan``. It is integrated numerically on a grid
    that reaches rates where the likelihood is below ``exp(-GRID_LOG_DROP)``
    of its maximum, so the result does not depend on the prior bound as long
    as the bound lies above the grid.

    Parameters
    ----------
    counts
        Observed counts, shape ``(datasets, groups, bins)`` or
        ``(groups, bins)`` for one dataset. Non-integer values are allowed
        (Asimov data).
    b_hat
        Fixed background prediction, shape ``(groups, bins)``; positive in
        every used bin.
    signal_template
        Signal exposure per bin in kg yr, shape ``(groups, bins)``.
    use_bin
        Bins entering the likelihood, shape ``(groups, bins)``.
    quantiles
        Posterior probabilities in (0, 1), e.g. ``(0.9,)`` for the 90%
        credible upper limit.
    n_grid
        Points of the fine grid on which the posterior is integrated.
    n_coarse
        Points of the coarse grid that locates the posterior.
    n_sigma
        Initial coarse-grid end, in posterior widths above the linearized rate
        estimate. The grid is doubled until it is wide enough.
    template_rtol
        Bins with a template below this fraction of its maximum add only an
        R-independent constant to the likelihood and are skipped.
    max_doublings
        Maximum number of coarse-grid doublings.
    memory_bytes
        Approximate memory budget for one chunk of datasets.

    Returns
    -------
    np.ndarray
        Quantiles, shape ``(datasets, len(quantiles))``, in counts/(kg yr).

    Raises
    ------
    ValueError
        If the shapes do not match, no bin is used, ``b_hat`` is not positive
        in a used bin, or the template or counts are invalid.
    """
    b_hat = np.asarray(b_hat, dtype=float)
    template = np.asarray(signal_template, dtype=float)
    used = np.asarray(use_bin, dtype=bool)
    if b_hat.ndim != 2 or template.shape != b_hat.shape or used.shape != b_hat.shape:
        msg = "b_hat, signal_template and use_bin must share one 2D shape."
        raise ValueError(msg)
    observed = np.asarray(counts, dtype=float)
    if observed.ndim == 2:
        observed = observed[None]
    if observed.ndim != 3 or observed.shape[1:] != b_hat.shape:
        msg = (
            f"counts must have shape (datasets, *{b_hat.shape}) or {b_hat.shape}; "
            f"got {np.shape(counts)}."
        )
        raise ValueError(msg)
    probabilities = np.asarray(quantiles, dtype=float).ravel()
    if probabilities.size == 0 or np.any((probabilities <= 0) | (probabilities >= 1)):
        msg = "quantiles must be a nonempty sequence of values in (0, 1)."
        raise ValueError(msg)
    if n_grid < 2 or n_coarse < 3:
        msg = "n_grid must be at least 2 and n_coarse at least 3."
        raise ValueError(msg)
    if not used.any():
        msg = "At least one bin must enter the likelihood."
        raise ValueError(msg)

    b_used, s_used, n_used = b_hat[used], template[used], observed[:, used]
    if not np.all(np.isfinite(b_used)) or np.any(b_used <= 0):
        msg = "b_hat must be finite and positive in every used bin."
        raise ValueError(msg)
    if not np.all(np.isfinite(s_used)) or np.any(s_used < 0):
        msg = "signal_template must be finite and nonnegative."
        raise ValueError(msg)
    if not np.all(np.isfinite(n_used)) or np.any(n_used < 0):
        msg = "counts must be finite and nonnegative."
        raise ValueError(msg)
    template_total = float(s_used.sum())
    if template_total <= 0:
        msg = "The signal template is zero in every used bin."
        raise ValueError(msg)

    keep = s_used > template_rtol * s_used.max()
    b_keep, s_keep, n_keep = b_used[keep], s_used[keep], n_used[:, keep]

    # Linearized rate estimate and posterior width at R = 0; 1 / S_total is the
    # width of the zero-background posterior exp(-R S_total).
    fisher = float(np.sum(s_keep**2 / b_keep))
    rate_linear = (n_keep - b_keep) @ (s_keep / b_keep) / fisher
    width = max(1.0 / np.sqrt(fisher), 1.0 / template_total)
    rate_high = np.maximum(rate_linear, 0.0) + n_sigma * width

    chunk = max(1, int(memory_bytes // (8 * max(n_grid, n_coarse) * s_keep.size)))
    output = np.empty((len(observed), probabilities.size))
    for start in range(0, len(observed), chunk):
        rows = slice(start, start + chunk)
        output[rows] = _chunk_quantiles(
            n_keep[rows],
            b_keep,
            s_keep,
            template_total,
            rate_high[rows],
            probabilities,
            n_grid,
            n_coarse,
            max_doublings,
        )
    return output


def rate_upper_limits(
    counts: np.ndarray,
    b_hat: np.ndarray,
    signal_template: np.ndarray,
    use_bin: np.ndarray,
    *,
    credibility: float = 0.9,
    **grid_options: Any,
) -> np.ndarray:
    """One-sided credible upper limit on the rate for every dataset.

    Grid-posterior counterpart of the Stan fit; see
    ``rate_posterior_quantiles`` for the arguments. Returns shape
    ``(datasets,)``, in counts/(kg yr).
    """
    return rate_posterior_quantiles(
        counts, b_hat, signal_template, use_bin, (credibility,), **grid_options
    )[:, 0]

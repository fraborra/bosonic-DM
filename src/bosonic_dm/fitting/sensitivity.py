# Copyright (C) 2026 Francesco Borra
#

"""Expected exclusion sensitivity from background-only pseudo-experiments.

Each pseudo-experiment ``k`` is a full broad-range spectrum per group,

    n_kgi ~ Poisson(B_hat_gi),

drawn once and analysed at every DM mass exactly as the data would be: the
fit window of ``build_signal_window`` is cut out and the 90% credible upper
limit on the rate is computed with the grid posterior
(``rate_upper_limits``). Neighbouring masses share the same spectra, so they
fluctuate together as in real data (scheme A in ``OPEN_TODOS.md``).

The limits are conditional on the fixed background ``B_hat`` and the nominal
signal response; neither uncertainty is propagated.
"""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from bosonic_dm.fitting.signal_fit import (
    CONDITIONAL_LABEL,
    STAN_GRID_AGREEMENT_MCSE,
    build_signal_stan_data,
    rate_posterior_quantiles,
    run_signal_fit,
    suggest_rate_upper,
    summarize_signal_fit,
)
from bosonic_dm.fitting.signal_template import (
    BackgroundPrediction,
    GroupEfficiency,
    SignalWindow,
    build_signal_window,
)

if TYPE_CHECKING:
    import cmdstanpy

logger = logging.getLogger(__name__)

_EDGE_ATOL_KEV = 1e-9

# Quantiles of the limit distribution over pseudo-experiments: median, then the
# 68% and 95% central bands.
BAND_QUANTILES = {
    "r90_q025": 0.025,
    "r90_q16": 0.16,
    "r90_median": 0.5,
    "r90_q84": 0.84,
    "r90_q975": 0.975,
}

# Last SeedSequence entropy word of the bootstrap stream, so it never equals a
# pseudo-experiment stream ``[base_seed, toy]``.
_BOOTSTRAP_STREAM = 2**32 - 1

# Entropy word of the Stan sampler seeds; distinct from the toy and bootstrap
# streams.
_STAN_STREAM = 2**32 - 2

# Thresholds of the Stan-versus-grid agreement criteria (``agreement_checks``).
# Per mass, the mean pull must satisfy |mean| <= PULL_MEAN_NSIGMA *
# PULL_SD_EXPECTED / sqrt(n_passed), where PULL_SD_EXPECTED is the observed
# spread of the pull (about 1.5, see STAN_GRID_AGREEMENT_MCSE).
PULL_MEAN_NSIGMA = 3.0
PULL_SD_EXPECTED = 1.5
POOLED_PULL_SD_RANGE = (0.7, 2.5)
RATIO_NSIGMA = 3.0
# Pooled fraction of passed toys with |z| > STAN_GRID_AGREEMENT_MCSE; the
# expected value is about 0.008 for a pull sd of 1.5.
MAX_PULL_OUTLIER_FRACTION = 0.02
MAX_FAILED_FRACTION = 0.01

TOY_SCHEME = "A: shared full spectra, n ~ Poisson(B_hat), all masses on each"


@dataclass(frozen=True)
class ExpectedLimits:
    """Per-toy limits, per-mass summary and the settings that produced them."""

    per_toy: pl.DataFrame
    summary: pl.DataFrame
    settings: dict[str, Any] = field(default_factory=dict)


def stack_backgrounds(
    backgrounds: Mapping[str, BackgroundPrediction],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the common bin edges and ``B_hat`` stacked as ``(groups, bins)``.

    Raises
    ------
    ValueError
        If no background is given or the groups do not share bin edges.
    """
    if not backgrounds:
        msg = "At least one background prediction is required."
        raise ValueError(msg)
    edges = next(iter(backgrounds.values())).bin_edges
    for name, background in backgrounds.items():
        if background.bin_edges.shape != edges.shape or not np.allclose(
            background.bin_edges, edges, rtol=0, atol=_EDGE_ATOL_KEV
        ):
            msg = f"Bin edges of group {name!r} differ from the other groups."
            raise ValueError(msg)
    return edges, np.vstack([b.b_hat for b in backgrounds.values()])


def draw_background_toys(
    b_hat: np.ndarray, n_toys: int, base_seed: int, first_toy: int = 0
) -> np.ndarray:
    """Draw background-only pseudo-spectra ``n ~ Poisson(b_hat)``.

    Toy ``k`` uses its own stream ``SeedSequence([base_seed, k])``, so it is
    the same whether it is drawn alone or with any other toys.

    Parameters
    ----------
    b_hat
        Expected counts, any shape (usually ``(groups, bins)``).
    n_toys
        Number of pseudo-spectra.
    base_seed
        Nonnegative base seed.
    first_toy
        Index of the first toy.

    Returns
    -------
    np.ndarray
        Counts, shape ``(n_toys, *b_hat.shape)``, dtype ``int64``.
    """
    b_hat = np.asarray(b_hat, dtype=float)
    if not np.all(np.isfinite(b_hat)) or np.any(b_hat < 0):
        msg = "b_hat must be finite and nonnegative."
        raise ValueError(msg)
    if n_toys < 1 or first_toy < 0 or base_seed < 0:
        msg = "n_toys must be positive; first_toy and base_seed nonnegative."
        raise ValueError(msg)
    toys = np.empty((n_toys, *b_hat.shape), dtype=np.int64)
    for row, toy in enumerate(range(first_toy, first_toy + n_toys)):
        rng = np.random.default_rng(np.random.SeedSequence([base_seed, toy]))
        toys[row] = rng.poisson(b_hat)
    return toys


def window_slice(full_edges: np.ndarray, window_edges: np.ndarray) -> slice:
    """Bin slice of the full spectrum covered by a fit window.

    Raises
    ------
    ValueError
        If the window edges are not a contiguous subset of the full edges.
    """
    full_edges = np.asarray(full_edges, dtype=float)
    window_edges = np.asarray(window_edges, dtype=float)
    start = int(np.searchsorted(full_edges, window_edges[0] - _EDGE_ATOL_KEV))
    stop = start + len(window_edges) - 1
    if stop >= len(full_edges) or not np.allclose(
        full_edges[start : stop + 1], window_edges, rtol=0, atol=_EDGE_ATOL_KEV
    ):
        msg = "The window edges are not a contiguous subset of the full edges."
        raise ValueError(msg)
    return slice(start, stop)


def _window_bins(
    full_edges: np.ndarray, mass_keV: float, half_width_keV: float
) -> slice:
    """Bins entirely inside ``mass +- half_width``, as in ``build_signal_window``."""
    low, high = mass_keV - half_width_keV, mass_keV + half_width_keV
    inside = np.flatnonzero(
        (full_edges[:-1] >= low - _EDGE_ATOL_KEV)
        & (full_edges[1:] <= high + _EDGE_ATOL_KEV)
    )
    if inside.size == 0:
        return slice(0, 0)
    return slice(int(inside[0]), int(inside[-1]) + 1)


def _bootstrap_median_error(
    limits: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> float:
    """Bootstrap standard deviation of the median of ``limits``."""
    if n_bootstrap < 2 or limits.size < 2:
        return float("nan")
    resampled = rng.integers(0, limits.size, size=(n_bootstrap, limits.size))
    return float(np.median(limits[resampled], axis=1).std(ddof=1))


def _masked_row(mass_keV: float, n_bins: int) -> dict[str, Any]:
    row: dict[str, Any] = {"mass_keV": float(mass_keV)}
    row.update(dict.fromkeys(BAND_QUANTILES, float("nan")))
    row.update(
        {
            "r90_median_mcse": float("nan"),
            "r90_asimov": float("nan"),
            "n_bins": n_bins,
            "n_masked_bins": n_bins,
            "near_masked_line": True,
            "all_bins_masked": True,
            "signal_fraction_in_fit_bins": 0.0,
            "b_hat_fit_bins": 0.0,
            "signal_exposure_fit_bins_kg_yr": 0.0,
        }
    )
    return row


def _window_row(
    window: SignalWindow,
    limits: np.ndarray,
    asimov: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    used = window.stacked("use_bin").astype(bool)
    template = window.stacked("signal_template")
    row: dict[str, Any] = {"mass_keV": window.mass_keV}
    row.update(
        {name: float(np.quantile(limits, q)) for name, q in BAND_QUANTILES.items()}
    )
    row.update(
        {
            "r90_median_mcse": _bootstrap_median_error(limits, n_bootstrap, rng),
            "r90_asimov": asimov,
            "n_bins": int(used.shape[1]),
            # Window bins masked in at least one group.
            "n_masked_bins": int((~used).any(axis=0).sum()),
            "near_masked_line": bool((~used).any()),
            "all_bins_masked": False,
            "signal_fraction_in_fit_bins": float(template[used].sum() / template.sum()),
            "b_hat_fit_bins": float(window.stacked("b_hat")[used].sum()),
            "signal_exposure_fit_bins_kg_yr": float(template[used].sum()),
        }
    )
    return row


def expected_limits(
    masses_keV: Sequence[float],
    backgrounds: Mapping[str, BackgroundPrediction],
    *,
    eres_dict: Mapping,
    groups_dict: Mapping[str, Mapping[str, str | Mapping[str, str | Sequence[str]]]],
    group_efficiencies: Mapping[str, GroupEfficiency],
    n_toys: int = 1000,
    base_seed: int = 20261117,
    credibility: float = 0.9,
    half_width_keV: float = 12.5,
    exposure_key: str = "expo",
    n_bootstrap: int = 1000,
    toys: np.ndarray | None = None,
    log_every: int = 50,
    **grid_options: Any,
) -> ExpectedLimits:
    """Distribution of the credible upper limit under the background-only model.

    Parameters
    ----------
    masses_keV
        DM masses at which to compute the limits.
    backgrounds
        Fixed background prediction per group, all on the same bin edges.
    eres_dict, groups_dict, group_efficiencies, half_width_keV, exposure_key
        Passed to ``build_signal_window``.
    n_toys
        Number of pseudo-experiments, ignored when ``toys`` is given.
    base_seed
        Base seed of ``draw_background_toys`` and of the bootstrap.
    credibility
        Credibility of the upper limit.
    n_bootstrap
        Bootstrap resamples for the Monte Carlo error of the median.
    toys
        Optional pseudo-spectra, shape ``(toys, groups, bins)`` on the
        background bin edges, e.g. with an injected signal.
    log_every
        Log progress every this many masses.
    **grid_options
        Passed to ``rate_posterior_quantiles``.

    Returns
    -------
    ExpectedLimits
        ``per_toy`` with columns ``mass_keV``, ``toy``, ``r90``; ``summary``
        with one row per mass (band quantiles, bootstrap error of the median,
        Asimov limit, masking flags); and the settings. Masses whose window is
        masked in every group have NaN limits and ``all_bins_masked``.
    """
    edges, b_hat_full = stack_backgrounds(backgrounds)
    toys_given = toys is not None
    if toys is None:
        toys = draw_background_toys(b_hat_full, n_toys, base_seed)
    elif toys.ndim != 3 or toys.shape[1:] != b_hat_full.shape:
        msg = f"toys must have shape (toys, *{b_hat_full.shape}); got {toys.shape}."
        raise ValueError(msg)
    used_full = np.vstack([b.used_in_fit for b in backgrounds.values()])
    n_toys = len(toys)
    toy_index = np.arange(n_toys)

    per_toy: list[pl.DataFrame] = []
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for count, mass in enumerate(masses_keV, start=1):
        bins = _window_bins(edges, float(mass), half_width_keV)
        if bins.stop > bins.start and not used_full[:, bins].any():
            rows.append(_masked_row(float(mass), bins.stop - bins.start))
            per_toy.append(
                pl.DataFrame(
                    {
                        "mass_keV": np.full(n_toys, float(mass)),
                        "toy": toy_index,
                        "r90": np.full(n_toys, np.nan),
                    }
                )
            )
            continue
        window = build_signal_window(
            float(mass),
            backgrounds,
            eres_dict=eres_dict,
            groups_dict=groups_dict,
            group_efficiencies=group_efficiencies,
            half_width_keV=half_width_keV,
            exposure_key=exposure_key,
        )
        bins = window_slice(edges, window.bin_edges)
        b_hat = window.stacked("b_hat")
        template = window.stacked("signal_template")
        use_bin = window.stacked("use_bin")
        limits = rate_posterior_quantiles(
            toys[:, :, bins], b_hat, template, use_bin, (credibility,), **grid_options
        )[:, 0]
        asimov = float(
            rate_posterior_quantiles(
                b_hat, b_hat, template, use_bin, (credibility,), **grid_options
            )[0, 0]
        )
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [base_seed, round(float(mass) * 1000), _BOOTSTRAP_STREAM]
            )
        )
        rows.append(_window_row(window, limits, asimov, n_bootstrap, rng))
        per_toy.append(
            pl.DataFrame(
                {
                    "mass_keV": np.full(n_toys, window.mass_keV),
                    "toy": toy_index,
                    "r90": limits,
                }
            )
        )
        if log_every and count % log_every == 0:
            logger.info(
                "%d/%d masses done (%.0f s).",
                count,
                len(masses_keV),
                time.perf_counter() - started,
            )

    settings = {
        "interpretation": CONDITIONAL_LABEL,
        "rate_units": "counts / (kg yr)",
        "credibility": float(credibility),
        "toy_scheme": TOY_SCHEME,
        "n_toys": int(n_toys),
        "base_seed": int(base_seed),
        "toys_given": toys_given,
        "half_width_keV": float(half_width_keV),
        "exposure_key": exposure_key,
        "n_bootstrap": int(n_bootstrap),
        "grid_options": dict(grid_options),
        "bin_edges_keV": [float(edges[0]), float(edges[-1])],
        "groups": {
            name: {
                "passed_diagnostics": background.passed_diagnostics,
                "files_sha256": dict(background.files_sha256),
            }
            for name, background in backgrounds.items()
        },
        "runtime_s": time.perf_counter() - started,
    }
    return ExpectedLimits(
        per_toy=pl.concat(per_toy) if per_toy else pl.DataFrame(),
        summary=pl.DataFrame(rows),
        settings=settings,
    )


def stan_limits(
    window: SignalWindow,
    toy_counts: np.ndarray,
    toy_index: Sequence[int],
    model: cmdstanpy.CmdStanModel,
    *,
    base_seed: int,
    credibility: float = 0.9,
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 2000,
    adapt_delta: float = 0.9,
    temporary_root: str | Path | None = None,
) -> pl.DataFrame:
    """Stan credible upper limits of background-only toys in one window.

    Parameters
    ----------
    window
        Signal window at one mass.
    toy_counts
        Toy counts, shape (toys, groups, window bins).
    toy_index
        Index of each toy, used for the sampler seed and the output table.
    model
        Compiled signal model.
    base_seed
        Base seed; the Stan seed of each (mass, toy) is derived from it.
    credibility
        Credibility of the upper limit.
    chains, iter_warmup, iter_sampling, adapt_delta
        Sampler settings.
    temporary_root
        Directory for the temporary CmdStan output (default: system temp).

    Returns
    -------
    polars.DataFrame
        One row per toy: ``mass_keV``, ``toy``, ``r90_stan``,
        ``r90_stan_mcse``, ``passed``, ``n_divergent`` and ``limit_to_bound``.
        Failed fits have NaN limits and ``passed = False``.
    """
    toy_counts = np.asarray(toy_counts)
    if toy_counts.shape[0] != len(toy_index):
        msg = "toy_counts and toy_index must have the same length."
        raise ValueError(msg)
    mass = float(window.mass_keV)
    nan = float("nan")
    rows: list[dict[str, Any]] = []
    for counts, toy_value in zip(toy_counts, toy_index, strict=True):
        toy = int(toy_value)
        seed = int(
            np.random.SeedSequence(
                [base_seed, _STAN_STREAM, toy, round(mass * 1000)]
            ).generate_state(1)[0]
            % (2**31 - 1)
        )
        row: dict[str, Any] = {
            "mass_keV": mass,
            "toy": toy,
            "r90_stan": nan,
            "r90_stan_mcse": nan,
            "passed": False,
            "n_divergent": 0,
            "limit_to_bound": nan,
        }
        try:
            toy_window = window.with_counts(counts)
            data = build_signal_stan_data(toy_window, suggest_rate_upper(toy_window))
            with tempfile.TemporaryDirectory(dir=temporary_root) as tmp:
                fit = run_signal_fit(
                    model,
                    data,
                    seed=seed,
                    chains=chains,
                    iter_warmup=iter_warmup,
                    iter_sampling=iter_sampling,
                    adapt_delta=adapt_delta,
                    output_dir=tmp,
                )
                summary = summarize_signal_fit(fit, data, credibility=credibility)
            row.update(
                r90_stan=float(summary.upper_limit),
                r90_stan_mcse=float(summary.rate["upper_limit_mcse"]),
                passed=bool(summary.passed),
                n_divergent=int(summary.diagnostics["divergences"]),
                limit_to_bound=float(summary.diagnostics["upper_limit_to_prior_bound"]),
            )
        except Exception:
            logger.warning(
                "Stan fit failed at mass %s keV, toy %d.", mass, toy, exc_info=True
            )
        rows.append(row)
    return pl.DataFrame(
        rows,
        schema={
            "mass_keV": pl.Float64,
            "toy": pl.Int64,
            "r90_stan": pl.Float64,
            "r90_stan_mcse": pl.Float64,
            "passed": pl.Boolean,
            "n_divergent": pl.Int64,
            "limit_to_bound": pl.Float64,
        },
    )


_COMPARISON_SCHEMA = {
    "mass_keV": pl.Float64,
    "near_masked_line": pl.Boolean,
    "n_toys": pl.Int64,
    "n_failed": pl.Int64,
    "r90_grid_median": pl.Float64,
    "r90_stan_median": pl.Float64,
    "median_ratio": pl.Float64,
    "median_ratio_err": pl.Float64,
    "r90_grid_q16": pl.Float64,
    "r90_grid_q84": pl.Float64,
    "r90_stan_q16": pl.Float64,
    "r90_stan_q84": pl.Float64,
    "pull_mean": pl.Float64,
    "pull_sd": pl.Float64,
    "frac_pull_above_max": pl.Float64,
}


def _pulls(table: pl.DataFrame) -> pl.DataFrame:
    """Add the pull ``z`` of the Stan limit with respect to the grid limit."""
    return table.with_columns(
        ((pl.col("r90_stan") - pl.col("r90")) / pl.col("r90_stan_mcse")).alias("z")
    )


def compare_stan_grid(
    per_toy: pl.DataFrame, *, n_bootstrap: int = 1000, seed: int = 0
) -> pl.DataFrame:
    """Per-mass comparison of the Stan and the grid upper limits.

    Parameters
    ----------
    per_toy
        Columns ``mass_keV``, ``toy``, ``r90`` (grid), ``r90_stan``,
        ``r90_stan_mcse``, ``passed`` and optionally ``near_masked_line``.
    n_bootstrap
        Bootstrap resamples for the error of the median ratio.
    seed
        Seed of the bootstrap.

    Returns
    -------
    polars.DataFrame
        One row per mass, sorted by mass. The pull is
        ``(r90_stan - r90) / r90_stan_mcse`` over the toys that passed; all
        medians and the pull are over those toys only. Masses without a
        passed toy have NaN metrics.
    """
    if "near_masked_line" not in per_toy.columns:
        per_toy = per_toy.with_columns(pl.lit(False).alias("near_masked_line"))
    per_toy = _pulls(per_toy)
    rng = np.random.default_rng(np.random.SeedSequence([seed, _BOOTSTRAP_STREAM, 1]))
    nan = float("nan")
    rows: list[dict[str, Any]] = []
    for mass in sorted(per_toy["mass_keV"].unique().to_list()):
        sub = per_toy.filter(pl.col("mass_keV") == mass).sort("toy")
        ok = sub.filter(pl.col("passed"))
        row: dict[str, Any] = {
            "mass_keV": float(mass),
            "near_masked_line": bool(sub["near_masked_line"][0]),
            "n_toys": sub.height,
            "n_failed": sub.height - ok.height,
        }
        row.update(dict.fromkeys(list(_COMPARISON_SCHEMA)[4:], nan))
        if ok.height:
            grid = ok["r90"].to_numpy()
            stan = ok["r90_stan"].to_numpy()
            z = ok["z"].to_numpy()
            row.update(
                r90_grid_median=float(np.median(grid)),
                r90_stan_median=float(np.median(stan)),
                median_ratio=float(np.median(stan) / np.median(grid)),
                r90_grid_q16=float(np.quantile(grid, 0.16)),
                r90_grid_q84=float(np.quantile(grid, 0.84)),
                r90_stan_q16=float(np.quantile(stan, 0.16)),
                r90_stan_q84=float(np.quantile(stan, 0.84)),
                pull_mean=float(z.mean()),
                pull_sd=float(z.std(ddof=1)) if z.size > 1 else nan,
                frac_pull_above_max=float(
                    np.mean(np.abs(z) > STAN_GRID_AGREEMENT_MCSE)
                ),
            )
            if n_bootstrap >= 2 and grid.size >= 2:
                resampled = rng.integers(0, grid.size, size=(n_bootstrap, grid.size))
                ratios = np.median(stan[resampled], axis=1) / np.median(
                    grid[resampled], axis=1
                )
                row["median_ratio_err"] = float(ratios.std(ddof=1))
        rows.append(row)
    return pl.DataFrame(rows, schema=_COMPARISON_SCHEMA)


def agreement_checks(comparison: pl.DataFrame, per_toy: pl.DataFrame) -> dict[str, Any]:
    """Pooled statistics and pass/fail criteria of the Stan-grid agreement.

    Parameters
    ----------
    comparison
        Output of ``compare_stan_grid``.
    per_toy
        The per-toy table it was computed from.

    Returns
    -------
    dict
        Pooled pull mean, sd and fraction above the agreement bound, the
        failed-fit fraction, the largest per-mass ``|pull_mean|``, the number
        of masses whose median ratio differs from one by more than
        ``RATIO_NSIGMA`` sigma (masses without a ratio error are excluded and
        counted in ``n_masses_without_ratio_error``), the boolean criteria and ``passed_all``.
        All values are plain Python types.
    """
    nan = float("nan")
    z = _pulls(per_toy).filter(pl.col("passed"))["z"].to_numpy()
    z = z[np.isfinite(z)]
    n_total = per_toy.height
    n_failed = int(per_toy.filter(~pl.col("passed")).height)
    failed_fraction = n_failed / n_total if n_total else nan

    valid = comparison.filter(pl.col("pull_mean").is_not_nan())
    n_ok = valid["n_toys"].to_numpy() - valid["n_failed"].to_numpy()
    pull_mean = valid["pull_mean"].to_numpy()
    mean_limit = PULL_MEAN_NSIGMA * PULL_SD_EXPECTED / np.sqrt(np.maximum(n_ok, 1))
    ratio = valid["median_ratio"].to_numpy()
    ratio_err = valid["median_ratio_err"].to_numpy()
    has_error = np.isfinite(ratio_err)
    ratio_off = has_error & (np.abs(ratio - 1.0) > RATIO_NSIGMA * ratio_err)

    pooled_sd = float(z.std(ddof=1)) if z.size > 1 else nan
    low, high = POOLED_PULL_SD_RANGE
    outlier_fraction = (
        float(np.mean(np.abs(z) > STAN_GRID_AGREEMENT_MCSE)) if z.size else nan
    )
    checks = {
        "pull_mean_per_mass": bool(np.all(np.abs(pull_mean) <= mean_limit)),
        "pooled_pull_sd": bool(low <= pooled_sd <= high),
        "pull_outlier_fraction": bool(outlier_fraction <= MAX_PULL_OUTLIER_FRACTION),
        "median_ratio_within_nsigma": bool(not np.any(ratio_off)),
        "failed_fraction": bool(failed_fraction < MAX_FAILED_FRACTION),
    }
    return {
        "pooled_pull_mean": float(z.mean()) if z.size else nan,
        "pooled_pull_sd": pooled_sd,
        "pooled_frac_pull_above_max": outlier_fraction,
        "ratio_nsigma": float(RATIO_NSIGMA),
        "n_masses_without_ratio_error": int(np.sum(~has_error)),
        "failed_fraction": float(failed_fraction),
        "max_abs_pull_mean": float(np.max(np.abs(pull_mean)))
        if pull_mean.size
        else nan,
        "n_masses_ratio_off": int(np.sum(ratio_off)),
        **{f"criterion_{name}": ok for name, ok in checks.items()},
        "passed_all": bool(all(checks.values())),
    }

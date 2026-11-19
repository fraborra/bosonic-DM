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
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from bosonic_dm.fitting.signal_fit import CONDITIONAL_LABEL, rate_posterior_quantiles
from bosonic_dm.fitting.signal_template import (
    BackgroundPrediction,
    GroupEfficiency,
    SignalWindow,
    build_signal_window,
)

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

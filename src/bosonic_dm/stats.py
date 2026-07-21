# Copyright (C) 2025 Francesco Borra
#

"""Statistical utilities for bosonic-DM analysis."""

from __future__ import annotations

import numpy as np


def weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute the weighted mean ``sum(w * v) / sum(w)``.

    Parameters
    ----------
    values
        Array of values.
    weights
        Array of weights (same length as *values*).

    Returns
    -------
    float
        Weighted mean, or ``nan`` if the total weight is zero or arrays are empty.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    total_w = np.sum(weights)
    if total_w == 0 or len(values) == 0:
        return np.nan
    return float(np.sum(weights * values) / total_w)


def bayesian_efficiency(
    k: int,
    n: int,
    alpha0: float = 0.5,
    beta0: float = 0.5,
) -> tuple[float, float]:
    """Compute Bayesian estimate of binomial efficiency with a Beta conjugate prior.

    Parameters
    ----------
    k
        Number of successes (events in the FEP window).
    n
        Number of originating vertices generated inside the detector.
    alpha0
        First shape parameter of the Beta prior (default: Jeffreys' 0.5).
    beta0
        Second shape parameter of the Beta prior (default: Jeffreys' 0.5).

    Returns
    -------
    tuple[float, float]
        Posterior mean (ratio) and posterior standard deviation (ratio uncertainty).

    Raises
    ------
    ValueError
        If there are no trials, if either count is negative, or if successes
        exceed trials. Missing denominators are an analysis-availability state,
        not an efficiency measurement.
    """
    if n <= 0:
        msg = "The number of trials must be positive."
        raise ValueError(msg)
    if k < 0:
        msg = "The number of successes cannot be negative."
        raise ValueError(msg)
    if k > n:
        msg = f"The number of successes ({k}) cannot exceed trials ({n})."
        raise ValueError(msg)

    alpha = alpha0 + k
    beta = beta0 + n - k

    mean = alpha / (alpha + beta)
    var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

    return mean, float(np.sqrt(var))


def compute_weighted_uncertainty(
    w_arr: np.ndarray,
    vals_arr: np.ndarray,
    mean_val: float,
    s_arr: np.ndarray,
) -> float:
    """Compute the total weighted uncertainty including measurement and scatter components.

    Parameters
    ----------
    w_arr
        Array of weights.
    vals_arr
        Array of values.
    mean_val
        Weighted mean of the values.
    s_arr
        Array of uncertainties for each value.

    Returns
    -------
    float
        Total combined uncertainty (measurement ⊕ scatter).
    """
    total_w = np.sum(w_arr)
    if total_w == 0:
        return 0.0

    # measurement component: quadrature propagation through weighted average
    unc_meas = float(np.sqrt(np.sum((w_arr * s_arr) ** 2)) / total_w)

    # scatter component: weighted variance across entries
    var_w = float(np.sum(w_arr * (vals_arr - mean_val) ** 2) / total_w)
    n_eff = float(total_w**2 / np.sum(w_arr**2))
    unc_scatter = float(np.sqrt(var_w / n_eff)) if n_eff > 0 else 0.0

    return float(np.sqrt(unc_meas**2 + unc_scatter**2))

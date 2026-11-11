# Copyright (C) 2026 Francesco Borra
#

"""Background spectral shapes, including NumPy mirrors of ``background.stan``.

``primakoff_rosen``, ``linear_background`` and ``modified_beta_g_pdf`` are
densities in counts/keV (or 1/keV for the Ar39 shape); the ``*_cdf`` functions
follow the Stan ``functions`` block, so binned predictions can be rebuilt from
posterior draws without Stan.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import beta as beta_function
from scipy.stats import beta as beta_dist

M_E_KEV = 510.99895
Q_GE76_KEV = 2039.0


def primakoff_rosen(
    energy_keV: ArrayLike, N: float, Q: float = Q_GE76_KEV
) -> np.ndarray:
    """Primakoff-Rosen approximation of the two-neutrino double-beta spectrum.

    Parameters
    ----------
    energy_keV
        Summed electron kinetic energy in keV.
    N
        Integral of the density over ``[0, Q]`` (``N_2nu`` in Stan).
    Q
        Q-value in keV.

    Returns
    -------
    np.ndarray
        Density in counts/keV; zero outside ``[0, Q]``.
    """
    q = Q / M_E_KEV
    t = np.clip(np.asarray(energy_keV, dtype=float) / M_E_KEV, 0.0, q)

    polynomial = t**4 + 10 * t**3 + 40 * t**2 + 60 * t + 30
    normalization = q**7 * (1980 + 990 * q + 220 * q**2 + 22 * q**3 + q**4) / 2772

    return N * t * polynomial * (q - t) ** 5 / (M_E_KEV * normalization)


def primakoff_rosen_cdf(energy_keV: ArrayLike, Q: float = Q_GE76_KEV) -> np.ndarray:
    """Exact integral of the normalized Primakoff-Rosen density on ``[0, E]``.

    Mirrors ``primakoff_rosen_cdf`` in ``background.stan``: in ``z = E/Q`` the
    density is a mixture of ``Beta(k + 2, 6)`` terms, ``k = 0..4``.
    """
    q = Q / M_E_KEV
    z = np.clip(np.asarray(energy_keV, dtype=float) / Q, 0.0, 1.0)
    coefficients = (30, 60, 40, 10, 1)
    weights = np.array(
        [c * q**k * beta_function(k + 2.0, 6.0) for k, c in enumerate(coefficients)]
    )
    terms = [w * beta_dist.cdf(z, k + 2.0, 6.0) for k, w in enumerate(weights)]
    return np.sum(terms, axis=0) / weights.sum()


def linear_background(
    energy_keV: ArrayLike, c0: float, cQ: float, Q: float = Q_GE76_KEV
) -> np.ndarray:
    """Straight line on ``[0, Q]`` with values ``c0`` at 0 and ``cQ`` at ``Q``.

    Both coefficients are in counts/keV, so the line is nonnegative on
    ``[0, Q]`` whenever they are.
    """
    z = np.asarray(energy_keV, dtype=float) / Q
    return c0 * (1.0 - z) + cQ * z


def modified_beta_g_pdf(
    energy_keV: ArrayLike,
    mu: float,
    sigma: float,
    tau: float,
    base_alpha: float,
    base_beta: float,
    loc: float,
    scale: float,
) -> np.ndarray:
    """Density of the modified beta-G family used for the Ar39 shape.

    The base ``G`` is ``Beta(base_alpha, base_beta)`` on
    ``[loc, loc + scale]``; ``mu``, ``sigma`` and ``tau`` deform it. The
    result is normalized to 1 on the base support and is in 1/keV.
    """
    x = np.asarray(energy_keV, dtype=float)
    rv = beta_dist(base_alpha, base_beta, loc=loc, scale=scale)
    u = rv.cdf(x)  # G(x), dimensionless
    g = rv.pdf(x)  # g(x), 1/keV
    inside = (u > 0.0) & (u < 1.0)
    u = np.where(inside, u, 0.5)  # placeholder keeps powers finite off-support

    denominator = 1.0 - (1.0 - tau) * u
    density = (
        tau**mu
        * g
        * u ** (mu - 1.0)
        * (1.0 - u) ** (sigma - 1.0)
        / (beta_function(mu, sigma) * denominator ** (mu + sigma))
    )
    return np.where(inside, density, 0.0)


def modified_beta_g_cdf(
    energy_keV: ArrayLike,
    mu: float,
    sigma: float,
    tau: float,
    base_alpha: float,
    base_beta: float,
    loc: float,
    scale: float,
) -> np.ndarray:
    """CDF of :func:`modified_beta_g_pdf`, as in ``background.stan``.

    With ``u = G(x)``, the CDF is ``I_r(mu, sigma)`` where
    ``r = tau u / (1 - u + tau u)``.
    """
    u = beta_dist.cdf(
        np.asarray(energy_keV, dtype=float), base_alpha, base_beta, loc=loc, scale=scale
    )
    r = tau * u / (1.0 - u + tau * u)
    return beta_dist.cdf(np.clip(r, 0.0, 1.0), mu, sigma)

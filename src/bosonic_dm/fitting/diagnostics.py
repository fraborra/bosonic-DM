# Copyright (C) 2026 Francesco Borra
#

"""Sampler diagnostics shared by the background and signal Stan fits.

The screening targets follow ``stan/README.md``: no divergences, no
maximum-treedepth hits, E-BFMI of at least 0.3, ``r_hat`` below 1.01, and
bulk and tail effective sample sizes of at least 400.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import cmdstanpy


def e_bfmi(energy: np.ndarray) -> np.ndarray:
    """Return the E-BFMI per chain from an energy array of shape ``(draws, chains)``."""
    return np.sum(np.diff(energy, axis=0) ** 2, axis=0) / np.sum(
        (energy - energy.mean(axis=0)) ** 2, axis=0
    )


def sampler_diagnostics(
    fit: cmdstanpy.CmdStanMCMC,
    parameters: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Collect sampler and convergence diagnostics of a fit.

    Parameters
    ----------
    fit
        Completed CmdStanPy fit.
    parameters
        Per-parameter summaries, each containing ``r_hat``, ``ess_bulk``, and
        ``ess_tail``.

    Returns
    -------
    dict
        Chain configuration, divergences, maximum-treedepth hits, minimum
        E-BFMI, maximum ``r_hat``, and minimum bulk and tail ESS.
    """
    energy = fit.draws_pd(vars=["energy__"])["energy__"].to_numpy()
    energy = energy.reshape(fit.chains, -1).T
    divergences = np.asarray(fit.divergences, dtype=int)
    treedepths = np.asarray(
        fit.max_treedepths if fit.max_treedepths is not None else [0], dtype=int
    )
    return {
        "chains": int(fit.chains),
        "draws_per_chain": int(fit.num_draws_sampling),
        "divergences": int(divergences.sum()),
        "max_treedepth_hits": int(treedepths.sum()),
        "e_bfmi_min": float(e_bfmi(energy).min()),
        "r_hat_max": max(float(p["r_hat"]) for p in parameters.values()),
        "ess_bulk_min": min(float(p["ess_bulk"]) for p in parameters.values()),
        "ess_tail_min": min(float(p["ess_tail"]) for p in parameters.values()),
    }


def diagnostics_passed(
    diagnostics: Mapping[str, Any],
    *,
    r_hat_max: float = 1.01,
    ess_min: float = 400.0,
    e_bfmi_min: float = 0.3,
) -> bool:
    """Return whether ``sampler_diagnostics`` output meets the screening targets."""
    return bool(
        diagnostics["divergences"] == 0
        and diagnostics["max_treedepth_hits"] == 0
        and diagnostics["e_bfmi_min"] >= e_bfmi_min
        and diagnostics["r_hat_max"] < r_hat_max
        and min(diagnostics["ess_bulk_min"], diagnostics["ess_tail_min"]) >= ess_min
    )

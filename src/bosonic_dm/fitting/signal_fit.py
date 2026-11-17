# Copyright (C) 2026 Francesco Borra
#

"""Fixed-background Stan signal fit at one DM mass and its credible upper limit.

The background ``B_hat`` and the response inputs (efficiency, exposure,
resolution) are fixed at their nominal values. The reported limit is therefore
the one-sided credible upper limit on the signal rate *conditional on the fixed
background and the nominal response*; background and response uncertainties are
not propagated.
"""

from __future__ import annotations

from collections.abc import Mapping
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

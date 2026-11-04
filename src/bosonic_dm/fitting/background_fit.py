# Copyright (C) 2026 Francesco Borra
#

"""Broad-range Stan background fit and its frozen per-bin prediction.

The fixed background used by the signal fits, ``B_hat``, is the per-bin
posterior median of Stan's ``expected_counts``: a summary of the joint
posterior evaluated on every bin, including bins left out of the likelihood.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from bosonic_dm.fitting.diagnostics import diagnostics_passed, sampler_diagnostics
from bosonic_dm.yaml_io import read_yaml, write_yaml

logger = logging.getLogger(__name__)

USABILITY_REMINDER = (
    "Check before trusting this fit: the group exposure counts only "
    "detector/runs with usability 'on', so events from 'ac' or 'off' "
    "detector/runs must already be excluded from the fitted spectrum "
    "(use bosonic_dm.cuts.select_usable / prepare_background_spectra(usable=...)). "
    "Otherwise counts enter without "
    "matching exposure."
)

if TYPE_CHECKING:
    import cmdstanpy

STAN_FILE = Path(__file__).resolve().parents[3] / "stan" / "background.stan"

RATE_PARAMETERS = ("N_2nu", "N_ar39", "c0", "cQ")
SHAPE_PARAMETERS = ("mu", "sigma", "tau", "base_alpha", "base_beta")
BACKGROUND_PARAMETERS = RATE_PARAMETERS + SHAPE_PARAMETERS
COMPONENTS = ("expected_2nu", "expected_linear", "expected_ar39")


@dataclass(frozen=True)
class BackgroundFitSummary:
    """Posterior summary, diagnostics, and fixed background of one group fit."""

    parameters: dict[str, dict[str, float]]
    diagnostics: dict[str, Any]
    passed: bool
    bin_edges: np.ndarray
    used_in_fit: np.ndarray
    b_hat: np.ndarray
    component_medians: dict[str, np.ndarray]
    expected_counts_draws: np.ndarray
    # Observed counts per bin; optional so summaries can be built without data.
    counts: np.ndarray | None = None


def _prefit_values(prefit: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return validated prefit rates and Ar39 shapes in Stan parameter order."""
    c0, c_q = prefit["linear_parameters"]
    rates = np.array([prefit["N_2nu"], prefit["N_ar39"], c0, c_q], dtype=float)
    shapes = np.asarray(prefit["ar39_parameters"], dtype=float)
    if shapes.shape != (len(SHAPE_PARAMETERS),):
        msg = f"ar39_parameters must have {len(SHAPE_PARAMETERS)} entries."
        raise ValueError(msg)
    if not (np.all(rates > 0) and np.all(shapes > 0)):
        msg = "Prefit values must be strictly positive (lognormal priors, positive inits)."
        raise ValueError(msg)
    return rates, shapes


def inits_from_prefit(
    prefit: Mapping[str, Any],
    *,
    chains: int = 4,
    jitter: float = 0.1,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Per-chain initial values at the prefit, with multiplicative jitter.

    Each parameter starts at ``value * exp(jitter * z)`` with ``z ~ N(0, 1)``
    drawn independently per chain, so chains start near the prefit but not at
    the same point (needed for a meaningful R-hat). ``jitter = 0`` starts every
    chain exactly at the prefit.
    """
    if chains < 1:
        msg = "chains must be at least 1."
        raise ValueError(msg)
    if not (np.isfinite(jitter) and jitter >= 0):
        msg = "jitter must be finite and nonnegative."
        raise ValueError(msg)
    values = np.concatenate(_prefit_values(prefit))
    rng = np.random.default_rng(seed)
    starts = values * np.exp(jitter * rng.standard_normal((chains, len(values))))
    return [
        dict(zip(BACKGROUND_PARAMETERS, map(float, start), strict=True))
        for start in starts
    ]


def priors_from_prefit(
    prefit: Mapping[str, Any],
    *,
    rate_sd: float = 1.0,
    shape_sd: float = 0.5,
) -> dict[str, np.ndarray]:
    """Centre lognormal priors on a curve_fit prefit, read by key name.

    Parameters
    ----------
    prefit
        One group's ``fit_ar`` entry with ``N_2nu``, ``N_ar39``,
        ``linear_parameters`` (``c0, cQ``), and ``ar39_parameters``
        (``mu, sigma, tau, base_alpha, base_beta``). Values are in counts,
        as fitted on ``counts_analysis``.
    rate_sd, shape_sd
        Prior standard deviations in log space for rates and Ar39 shapes.
    """
    rates, shapes = _prefit_values(prefit)
    if not (rate_sd > 0 and shape_sd > 0):
        msg = "Prior standard deviations must be strictly positive."
        raise ValueError(msg)
    return {
        "log_rate_prior_mean": np.log(rates),
        "log_rate_prior_sd": np.full(len(RATE_PARAMETERS), float(rate_sd)),
        "log_shape_prior_mean": np.log(shapes),
        "log_shape_prior_sd": np.full(len(SHAPE_PARAMETERS), float(shape_sd)),
    }


def build_stan_data(
    spectrum: Mapping[str, Any],
    priors: Mapping[str, Any],
    *,
    counts_key: str = "counts_analysis",
    q_value_keV: float = 2039.0,
    ar39_loc_keV: float = 0.0,
    ar39_scale_keV: float = 565.8,
) -> dict[str, Any]:
    """Build ``background.stan`` data from a prepared group spectrum.

    ``spectrum`` is a ``prepare_background_spectra`` or ``spectrum_in_range``
    group. Its ``fit_mask`` becomes ``use_bin``, so peak windows keep their
    real counts but are left out of the likelihood.
    """
    edges = np.asarray(spectrum["bin_edges"], dtype=float)
    counts = np.asarray(spectrum[counts_key])
    use_bin = np.asarray(spectrum["fit_mask"], dtype=bool)
    if counts.shape != use_bin.shape or len(edges) != len(counts) + 1:
        msg = "Counts, fit_mask, and bin_edges have inconsistent lengths."
        raise ValueError(msg)
    if np.any(np.diff(edges) <= 0) or edges[0] < 0 or edges[-1] > q_value_keV:
        msg = "Bin edges must increase strictly and lie within [0, Q]."
        raise ValueError(msg)
    if not use_bin.any():
        msg = "At least one bin must be used in the fit."
        raise ValueError(msg)
    if np.any(counts < 0) or not np.all(np.equal(np.mod(counts, 1), 0)):
        msg = "Counts must be nonnegative integers."
        raise ValueError(msg)
    return {
        "n_bins": len(counts),
        "edges": edges,
        "counts": counts.astype(int),
        "use_bin": use_bin.astype(int),
        "Q": float(q_value_keV),
        "loc": float(ar39_loc_keV),
        "scale": float(ar39_scale_keV),
        **{
            key: np.asarray(priors[key], dtype=float)
            for key in (
                "log_rate_prior_mean",
                "log_rate_prior_sd",
                "log_shape_prior_mean",
                "log_shape_prior_sd",
            )
        },
    }


def run_background_fit(
    model: cmdstanpy.CmdStanModel,
    stan_data: Mapping[str, Any],
    *,
    seed: int,
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 1000,
    adapt_delta: float = 0.9,
    output_dir: str | Path | None = None,
    show_progress: bool = False,
    inits: Mapping[str, float] | Sequence[Mapping[str, float]] | None = None,
) -> cmdstanpy.CmdStanMCMC:
    """Sample the background posterior for one detector group.

    ``inits`` is optional: one mapping for all chains, one mapping per chain
    (e.g. from ``inits_from_prefit``), or ``None`` for Stan's default random
    initialization.
    """
    logger.warning(USABILITY_REMINDER)
    if inits is not None and not isinstance(inits, Mapping):
        inits = [dict(init) for init in inits]
        if len(inits) != chains:
            msg = f"Got {len(inits)} per-chain inits for {chains} chains."
            raise ValueError(msg)
    elif inits is not None:
        inits = dict(inits)
    return model.sample(
        data=dict(stan_data),
        inits=inits,
        chains=chains,
        parallel_chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        adapt_delta=adapt_delta,
        seed=seed,
        output_dir=None if output_dir is None else str(output_dir),
        show_progress=show_progress,
    )


def to_inference_data(fit: cmdstanpy.CmdStanMCMC, stan_data: Mapping[str, Any]) -> Any:
    """Convert a fit to ArviZ, pairing ``log_lik`` with the fitted bins only."""
    import arviz as az  # noqa: PLC0415

    use_bin = np.asarray(stan_data["use_bin"], dtype=bool)
    bin_variables = ("counts_rep", "expected_counts", *COMPONENTS)
    return az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive=["counts_rep"],
        observed_data={"counts": np.asarray(stan_data["counts"])[use_bin]},
        log_likelihood={"counts": "log_lik"},
        coords={
            "energy_bin": np.arange(stan_data["n_bins"]),
            "fit_bin": np.flatnonzero(use_bin),
        },
        dims={
            **{name: ["energy_bin"] for name in bin_variables},
            "counts": ["fit_bin"],
            "log_lik": ["fit_bin"],
        },
    )


def summarize_background_fit(
    fit: cmdstanpy.CmdStanMCMC,
    stan_data: Mapping[str, Any],
    *,
    n_saved_draws: int = 200,
    seed: int = 0,
    r_hat_max: float = 1.01,
    ess_min: float = 400.0,
    e_bfmi_min: float = 0.3,
) -> BackgroundFitSummary:
    """Summarize parameters and diagnostics, and freeze ``B_hat``.

    The pass flag uses the screening targets in ``stan/README.md``. ``B_hat``
    is the per-bin posterior median of ``expected_counts`` over all bins.
    A random thinned subset of the ``expected_counts`` draws is kept for
    background-uncertainty studies.
    """
    import arviz as az  # noqa: PLC0415

    table = az.summary(
        to_inference_data(fit, stan_data),
        var_names=list(BACKGROUND_PARAMETERS),
        ci_prob=0.95,
        ci_kind="eti",
        round_to="none",
    )
    parameters = {}
    for name in BACKGROUND_PARAMETERS:
        draws = fit.stan_variable(name)
        parameters[name] = {
            "median": float(np.median(draws)),
            "eti95_lb": float(table.loc[name, "eti95_lb"]),
            "eti95_ub": float(table.loc[name, "eti95_ub"]),
            "r_hat": float(table.loc[name, "r_hat"]),
            "ess_bulk": float(table.loc[name, "ess_bulk"]),
            "ess_tail": float(table.loc[name, "ess_tail"]),
        }

    diagnostics = sampler_diagnostics(fit, parameters)
    passed = diagnostics_passed(
        diagnostics, r_hat_max=r_hat_max, ess_min=ess_min, e_bfmi_min=e_bfmi_min
    )

    expected = fit.stan_variable("expected_counts")
    rng = np.random.default_rng(seed)
    keep = np.sort(
        rng.choice(len(expected), size=min(n_saved_draws, len(expected)), replace=False)
    )
    return BackgroundFitSummary(
        parameters=parameters,
        diagnostics=diagnostics,
        passed=bool(passed),
        bin_edges=np.asarray(stan_data["edges"], dtype=float),
        counts=np.asarray(stan_data["counts"], dtype=int),
        used_in_fit=np.asarray(stan_data["use_bin"], dtype=bool),
        b_hat=np.median(expected, axis=0),
        component_medians={
            name: np.median(fit.stan_variable(name), axis=0) for name in COMPONENTS
        },
        expected_counts_draws=expected[keep],
    )


def _excluded_intervals(edges: np.ndarray, used: np.ndarray) -> list[list[float]]:
    """Merge consecutive unused bins into ``[low, high]`` energy intervals."""
    intervals: list[list[float]] = []
    for b in np.flatnonzero(~used):
        if intervals and intervals[-1][1] == edges[b]:
            intervals[-1][1] = float(edges[b + 1])
        else:
            intervals.append([float(edges[b]), float(edges[b + 1])])
    return intervals


def save_background_fit(
    fit: cmdstanpy.CmdStanMCMC,
    summary: BackgroundFitSummary,
    stan_data: Mapping[str, Any],
    output_dir: str | Path,
    *,
    provenance: Mapping[str, Any] | None = None,
    stan_file: str | Path = STAN_FILE,
) -> dict[str, Path]:
    """Write the posterior, ``B_hat`` table, thinned draws, and a YAML summary.

    Files written in ``output_dir``: ``posterior.nc`` (ArviZ),
    ``bhat.parquet`` (bin edges, ``B_hat``, ``used_in_fit``, component
    medians), ``expected_counts_draws.npy`` (thinned draws, shape
    ``(draws, bins)``), and ``summary.yaml``. ``provenance`` is copied into the
    YAML summary, e.g. group name, input dataset, peak list and window.
    """
    import cmdstanpy  # noqa: PLC0415

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "posterior": output / "posterior.nc",
        "bhat": output / "bhat.parquet",
        "draws": output / "expected_counts_draws.npy",
        "summary": output / "summary.yaml",
    }

    to_inference_data(fit, stan_data).to_netcdf(paths["posterior"])
    edges = summary.bin_edges
    pl.DataFrame(
        {
            "bin_lo_keV": edges[:-1],
            "bin_hi_keV": edges[1:],
            "counts": np.asarray(stan_data["counts"]),
            "used_in_fit": summary.used_in_fit,
            "B_hat": summary.b_hat,
            **{
                f"{name}_median": values
                for name, values in summary.component_medians.items()
            },
        }
    ).write_parquet(paths["bhat"])
    np.save(paths["draws"], summary.expected_counts_draws)

    config = fit.metadata.cmdstan_config
    write_yaml(
        {
            "b_hat_rule": "per-bin posterior median of expected_counts",
            "passed_diagnostics": summary.passed,
            "diagnostics": summary.diagnostics,
            "parameters": summary.parameters,
            "fit_range_keV": [float(edges[0]), float(edges[-1])],
            "n_bins": int(stan_data["n_bins"]),
            "n_fit_bins": int(summary.used_in_fit.sum()),
            "excluded_intervals_keV": _excluded_intervals(edges, summary.used_in_fit),
            "fixed_inputs": {
                "Q_keV": stan_data["Q"],
                "ar39_loc_keV": stan_data["loc"],
                "ar39_scale_keV": stan_data["scale"],
            },
            "priors": {
                key: stan_data[key]
                for key in (
                    "log_rate_prior_mean",
                    "log_rate_prior_sd",
                    "log_shape_prior_mean",
                    "log_shape_prior_sd",
                )
            },
            "sampler": {
                "seed": config.get("seed"),
                "num_warmup": config.get("num_warmup"),
                "num_samples": config.get("num_samples"),
                "adapt_delta": config.get("delta"),
            },
            "software": {
                "stan_file": Path(stan_file).name,
                "stan_file_sha256": hashlib.sha256(
                    Path(stan_file).read_bytes()
                ).hexdigest(),
                "cmdstan_version": ".".join(
                    map(str, cmdstanpy.cmdstan_version() or ())
                ),
                "cmdstanpy_version": cmdstanpy.__version__,
            },
            "provenance": dict(provenance or {}),
            "files": {name: path.name for name, path in paths.items()},
        },
        paths["summary"],
    )
    return paths


def load_background_fit(output_dir: str | Path) -> BackgroundFitSummary:
    """Rebuild a ``BackgroundFitSummary`` from ``save_background_fit`` files.

    Reads ``summary.yaml``, ``bhat.parquet``, and ``expected_counts_draws.npy``
    in ``output_dir``. The full posterior stays in ``posterior.nc``; open it
    with ``xarray.open_datatree`` when needed.
    """
    output = Path(output_dir)
    record = read_yaml(output / "summary.yaml")
    table = pl.read_parquet(output / "bhat.parquet")
    lo, hi = table["bin_lo_keV"].to_numpy(), table["bin_hi_keV"].to_numpy()
    return BackgroundFitSummary(
        parameters=record["parameters"],
        diagnostics=record["diagnostics"],
        passed=bool(record["passed_diagnostics"]),
        bin_edges=np.append(lo, hi[-1]),
        counts=table["counts"].to_numpy(),
        used_in_fit=table["used_in_fit"].to_numpy(),
        b_hat=table["B_hat"].to_numpy(),
        component_medians={
            name: table[f"{name}_median"].to_numpy() for name in COMPONENTS
        },
        expected_counts_draws=np.load(output / "expected_counts_draws.npy"),
    )

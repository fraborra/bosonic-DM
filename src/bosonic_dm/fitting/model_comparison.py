# Copyright (C) 2026 Francesco Borra
#

"""Bayes factors and LOO comparisons between background-model variants.

Marginal likelihoods are estimated by bridge sampling (Meng & Wong 1996),
following the iterative scheme and normal proposal of Gronau et al. (2017),
the algorithm of the R ``bridgesampling`` package and of BAT.jl's
``BridgeSampling``. Posterior draws come from CmdStan; the log posterior is
evaluated by ``background_log_density`` on the unconstrained (log) scale.

The relative mean-squared error of the estimate follows Frühwirth-Schnatter
(2004), with the posterior term divided by the effective sample size instead
of the raw number of draws, so autocorrelated MCMC draws are accounted for.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from arviz_stats.base import array_stats
from scipy.special import expit, logsumexp
from scipy.stats import multivariate_normal

from bosonic_dm.fitting.background_fit import (
    BACKGROUND_PARAMETERS,
    background_log_density,
    fixed_parameters,
    free_parameters,
    to_inference_data,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# Inputs that must agree for two fits to describe the same spectrum.
_SHARED_DATA_KEYS = ("n_bins", "edges", "counts", "use_bin", "Q", "loc", "scale")

# Kass & Raftery (1995), applied to 2 ln BF.
_KASS_RAFTERY = (
    (2.0, "not worth more than a bare mention"),
    (6.0, "positive"),
    (10.0, "strong"),
    (np.inf, "very strong"),
)


@dataclass(frozen=True)
class BridgeSamplingResult:
    """Bridge-sampling estimate of a log marginal likelihood."""

    log_ml: float
    # Approximate standard deviation of log_ml (square root of the relative MSE).
    log_ml_sd: float
    n_iterations: int
    converged: bool
    # Posterior draws used in the iterative scheme, and their effective number.
    n_posterior: int
    n_effective: float
    n_proposal: int


def _iterate(
    l1: np.ndarray,
    l2: np.ndarray,
    n_effective: float,
    *,
    tol: float,
    max_iter: int,
) -> tuple[float, int, bool]:
    """Run the Meng-Wong fixed-point iteration in log space.

    ``l1`` and ``l2`` are ``log q - log g`` at posterior and proposal draws.
    Returns ``log_ml``, the number of iterations, and whether it converged.
    """
    lstar = float(np.median(l1))
    n1, n2 = len(l1), len(l2)
    log_s1 = np.log(n_effective / (n_effective + n2))
    log_s2 = np.log(n2 / (n_effective + n2))
    a1, a2 = l1 - lstar, l2 - lstar
    log_r = 0.0
    for iteration in range(1, max_iter + 1):
        log_num = a2 - np.logaddexp(log_s1 + a2, log_s2 + log_r)
        log_den = -np.logaddexp(log_s1 + a1, log_s2 + log_r)
        new = (logsumexp(log_num) - np.log(n2)) - (logsumexp(log_den) - np.log(n1))
        if not np.isfinite(new):
            msg = "Bridge sampling iteration produced a non-finite value."
            raise FloatingPointError(msg)
        if abs(new - log_r) < tol:
            return new + lstar, iteration, True
        log_r = new
    return log_r + lstar, max_iter, False


@dataclass(frozen=True)
class BridgeSamplingTerms:
    """Log densities that enter a bridge-sampling estimate.

    ``q`` is the unnormalized log posterior and ``g`` the normal proposal.
    The ``*_posterior`` arrays are evaluated at the posterior draws of the
    iterative scheme (second half of each chain, chain-major), the
    ``*_proposal`` arrays at the proposal draws. For a reliable estimate the
    distributions of ``log q - log g`` at the two sets of draws overlap.
    """

    log_q_posterior: np.ndarray
    log_g_posterior: np.ndarray
    log_q_proposal: np.ndarray
    log_g_proposal: np.ndarray
    # Second half of each chain, shape (chains, draws // 2, dim).
    posterior_draws: np.ndarray
    proposal_mean: np.ndarray
    proposal_cov: np.ndarray
    n_effective: float


def _validated_draws(draws: np.ndarray) -> np.ndarray:
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 3:
        msg = "draws must have shape (chains, draws, dim)."
        raise ValueError(msg)
    if draws.shape[1] < 4:
        msg = "Need at least 4 draws per chain."
        raise ValueError(msg)
    return draws


def bridge_sampling_terms(
    log_density: Callable[[np.ndarray], np.ndarray],
    draws: np.ndarray,
    *,
    seed: int = 0,
    log_q_posterior: np.ndarray | None = None,
) -> BridgeSamplingTerms:
    """Fit the proposal and evaluate both densities at both sets of draws.

    Arguments are those of :func:`bridge_sampling`. ``log_q_posterior``, if
    given, replaces the evaluation of ``log_density`` at the posterior draws,
    e.g. to repeat the estimate with another proposal ``seed`` at the cost of
    the proposal draws only.
    """
    draws = _validated_draws(draws)
    _, n_draws, dim = draws.shape
    half = n_draws // 2
    fit_part = draws[:, :half].reshape(-1, dim)
    iter_part = draws[:, half:]
    posterior = iter_part.reshape(-1, dim)

    mean = fit_part.mean(axis=0)
    cov = np.atleast_2d(np.cov(fit_part, rowvar=False))
    proposal = multivariate_normal(mean=mean, cov=cov)
    n2 = len(posterior)
    proposal_draws = proposal.rvs(size=n2, random_state=np.random.default_rng(seed))
    proposal_draws = np.reshape(proposal_draws, (n2, dim))

    if log_q_posterior is None:
        q11 = np.asarray(log_density(posterior), dtype=float)
    else:
        q11 = np.asarray(log_q_posterior, dtype=float)
        if q11.shape != (n2,):
            msg = f"log_q_posterior must have shape ({n2},)."
            raise ValueError(msg)
    q12 = np.atleast_1d(proposal.logpdf(posterior))
    q21 = np.asarray(log_density(proposal_draws), dtype=float)
    q22 = np.atleast_1d(proposal.logpdf(proposal_draws))
    if not np.all(np.isfinite(q11)):
        msg = "log_density is not finite at every posterior draw."
        raise ValueError(msg)
    # Proposal draws far in the tails may underflow; they carry no weight.
    q21 = np.where(np.isfinite(q21), q21, -np.inf)

    ess = np.atleast_1d(
        array_stats.ess(iter_part, chain_axis=0, draw_axis=1, method="bulk")
    )
    return BridgeSamplingTerms(
        log_q_posterior=q11,
        log_g_posterior=q12,
        log_q_proposal=q21,
        log_g_proposal=q22,
        posterior_draws=iter_part,
        proposal_mean=mean,
        proposal_cov=cov,
        n_effective=float(min(np.median(ess), len(posterior))),
    )


def bridge_sampling_estimate(
    terms: BridgeSamplingTerms,
    *,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> BridgeSamplingResult:
    """Run the iterative scheme and its error estimate on ``terms``."""
    q11, q12 = terms.log_q_posterior, terms.log_g_posterior
    q21, q22 = terms.log_q_proposal, terms.log_g_proposal
    n_chains = terms.posterior_draws.shape[0]
    n1, n2 = len(q11), len(q21)
    n_effective = terms.n_effective
    log_ml, n_iterations, converged = _iterate(
        q11 - q12, q21 - q22, n_effective, tol=tol, max_iter=max_iter
    )
    if not converged:
        logger.warning("Bridge sampling did not converge in %d iterations.", max_iter)

    # Relative MSE (Frühwirth-Schnatter 2004), with s1, s2 from the raw counts.
    log_s1, log_s2 = np.log(n1 / (n1 + n2)), np.log(n2 / (n1 + n2))
    log_p_proposal = q21 - log_ml
    log_p_posterior = q11 - log_ml
    f1 = np.exp(log_p_proposal - np.logaddexp(log_s1 + log_p_proposal, log_s2 + q22))
    f2 = np.exp(q12 - np.logaddexp(log_s1 + log_p_posterior, log_s2 + q12))
    f2_ess = float(
        min(
            np.squeeze(
                array_stats.ess(
                    f2.reshape(n_chains, -1), chain_axis=0, draw_axis=1, method="bulk"
                )
            ),
            n1,
        )
    )
    re2 = np.var(f1) / (n2 * np.mean(f1) ** 2) + np.var(f2) / (
        f2_ess * np.mean(f2) ** 2
    )
    return BridgeSamplingResult(
        log_ml=float(log_ml),
        log_ml_sd=float(np.sqrt(re2)),
        n_iterations=int(n_iterations),
        converged=bool(converged),
        n_posterior=int(n1),
        n_effective=n_effective,
        n_proposal=int(n2),
    )


def bridge_sampling(
    log_density: Callable[[np.ndarray], np.ndarray],
    draws: np.ndarray,
    *,
    seed: int = 0,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> BridgeSamplingResult:
    """Estimate ``log ∫ q(θ) dθ`` from posterior draws by bridge sampling.

    Parameters
    ----------
    log_density
        Unnormalized log posterior ``log q`` on the same (unconstrained)
        scale as ``draws``, including all constants and any Jacobian. It maps
        an ``(n, dim)`` array to ``(n,)``.
    draws
        Posterior draws on the unconstrained scale, shape
        ``(chains, draws, dim)``. In each chain the first half fits the
        multivariate-normal proposal and the second half enters the iterative
        scheme, as in the R ``bridgesampling`` package.
    seed
        Seed of the proposal draws.
    tol, max_iter
        Convergence tolerance on ``log r`` and maximum number of iterations.

    Returns
    -------
    BridgeSamplingResult
        ``log_ml`` and its approximate standard deviation.
    """
    terms = bridge_sampling_terms(log_density, draws, seed=seed)
    return bridge_sampling_estimate(terms, tol=tol, max_iter=max_iter)


def _is_cmdstan_fit(posterior: Any) -> bool:
    return hasattr(posterior, "draws") and hasattr(posterior, "column_names")


def _parameter_draws(posterior: Any, names: Sequence[str]) -> np.ndarray:
    """Draws of ``names``, shaped ``(chains, draws, len(names))``.

    ``posterior`` is a ``cmdstanpy.CmdStanMCMC`` (CmdStan column names) or an
    ArviZ object with a ``posterior`` group, such as a ``posterior.nc`` from
    ``save_background_fit`` opened with ``xarray.open_datatree``. There,
    ``lp__`` is read from ``sample_stats.lp``.
    """
    if _is_cmdstan_fit(posterior):
        columns = list(posterior.column_names)
        draws = posterior.draws(concat_chains=False)  # (draws, chains, columns)
        return np.moveaxis(draws[:, :, [columns.index(name) for name in names]], 0, 1)
    arrays = []
    for name in names:
        if name == "lp__":
            values = posterior["sample_stats"]["lp"]
        else:
            values = posterior["posterior"][name]
        arrays.append(np.asarray(values.transpose("chain", "draw").values, dtype=float))
    return np.stack(arrays, axis=-1)


def background_unconstrained_draws(
    posterior: Any,
    stan_data: Mapping[str, Any],
    *,
    seed: int = 0,
    n_checked: int = 200,
    lp_atol: float = 0.05,
    lp_rtol: float = 1e-5,
) -> np.ndarray:
    """Log of the free-parameter draws, checked against Stan's ``lp__``.

    Returns shape ``(chains, draws, len(free_parameters(stan_data)))``.
    ``posterior`` is a ``CmdStanMCMC`` or a reloaded ``posterior.nc``. The
    NumPy ``background_log_density`` is compared with ``lp__`` on
    ``n_checked`` random draws; a mismatch beyond
    ``lp_atol + lp_rtol * |lp__|`` raises. This catches fits from a model
    whose ``lp__`` drops constants (``~`` statements) or ``stan_data`` that
    differs from the fitted data. The tolerance allows for the precision of
    the CmdStan output (6 significant figures by default).
    """
    names = list(free_parameters(stan_data))
    draws = _parameter_draws(posterior, [*names, "lp__"])
    theta = np.log(draws[..., :-1])
    lp = draws[..., -1].ravel()

    flat = theta.reshape(-1, len(names))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(flat), size=min(n_checked, len(flat)), replace=False)
    mismatch = np.abs(background_log_density(flat[idx], stan_data) - lp[idx])
    if np.any(mismatch > lp_atol + lp_rtol * np.abs(lp[idx])):
        msg = (
            "NumPy background_log_density disagrees with Stan lp__ "
            f"(max difference {mismatch.max():.3g}). Was the fit made with the "
            "current background.stan and this stan_data?"
        )
        raise ValueError(msg)
    return theta


def background_log_marginal_likelihood(
    posterior: Any,
    stan_data: Mapping[str, Any],
    *,
    seed: int = 0,
    n_checked: int = 200,
    lp_atol: float = 0.05,
    lp_rtol: float = 1e-5,
) -> BridgeSamplingResult:
    """Log marginal likelihood of one background fit by bridge sampling.

    ``posterior`` is a ``cmdstanpy.CmdStanMCMC`` or a reloaded
    ``posterior.nc``. The free parameters are taken on the log scale and
    checked against ``lp__`` first (see
    :func:`background_unconstrained_draws`).
    """
    theta = background_unconstrained_draws(
        posterior,
        stan_data,
        seed=seed,
        n_checked=n_checked,
        lp_atol=lp_atol,
        lp_rtol=lp_rtol,
    )
    return bridge_sampling(
        partial(background_log_density, stan_data=stan_data), theta, seed=seed
    )


@dataclass(frozen=True)
class BridgeSamplingCheck:
    """Bridge-sampling estimate of one fit, with its stability checks.

    ``by_seed`` repeats the estimate with other proposal seeds on the same
    draws; its spread should match ``result.log_ml_sd``. ``by_draws`` uses
    only the first ``n_draws`` draws of each chain; the estimate should
    settle as the number of draws grows.
    """

    parameters: tuple[str, ...]
    terms: BridgeSamplingTerms
    result: BridgeSamplingResult
    seeds: tuple[int, ...]
    by_seed: tuple[BridgeSamplingResult, ...]
    n_draws: tuple[int, ...]
    by_draws: tuple[BridgeSamplingResult, ...]


def check_bridge_sampling(
    posterior: Any,
    stan_data: Mapping[str, Any],
    *,
    seed: int = 0,
    seeds: Sequence[int] = tuple(range(1, 11)),
    n_draws: Sequence[int] | None = None,
) -> BridgeSamplingCheck:
    """Repeat the bridge-sampling estimate of one fit to test its stability.

    Parameters
    ----------
    posterior
        A ``cmdstanpy.CmdStanMCMC`` or a reloaded ``posterior.nc``.
    stan_data
        The fitted data, e.g. from ``load_stan_data``.
    seed
        Proposal seed of the reference estimate (as in
        :func:`compare_background_models`).
    seeds
        Further proposal seeds. Each reuses the log posterior at the
        posterior draws, so it costs one evaluation at the proposal draws.
    n_draws
        Draws per chain for the truncated estimates; ``None`` uses 1/8, 1/4,
        1/2 and all of them.
    """
    theta = background_unconstrained_draws(posterior, stan_data, seed=seed)
    log_density = partial(background_log_density, stan_data=stan_data)
    terms = bridge_sampling_terms(log_density, theta, seed=seed)
    total = theta.shape[1]
    if n_draws is None:
        n_draws = sorted({max(4, total // k) for k in (8, 4, 2, 1)})
    n_draws = tuple(int(n) for n in n_draws if 4 <= n <= total)
    return BridgeSamplingCheck(
        parameters=free_parameters(stan_data),
        terms=terms,
        result=bridge_sampling_estimate(terms),
        seeds=tuple(int(s) for s in seeds),
        by_seed=tuple(
            bridge_sampling_estimate(
                bridge_sampling_terms(
                    log_density,
                    theta,
                    seed=int(s),
                    log_q_posterior=terms.log_q_posterior,
                )
            )
            for s in seeds
        ),
        n_draws=n_draws,
        by_draws=tuple(
            bridge_sampling(log_density, theta[:, :n], seed=seed) for n in n_draws
        ),
    )


def model_label(stan_data: Mapping[str, Any]) -> str:
    """Short name of a background variant from its fixed parameters."""
    fixed = fixed_parameters(stan_data)
    if not fixed:
        return "full"
    if fixed == {"c0": 0.0, "cQ": 0.0}:
        return "no linear"
    return ", ".join(f"{name}={value:g}" for name, value in fixed.items())


def _prior_table(stan_data: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    means = np.concatenate(
        [stan_data["log_rate_prior_mean"], stan_data["log_shape_prior_mean"]]
    )
    sds = np.concatenate(
        [stan_data["log_rate_prior_sd"], stan_data["log_shape_prior_sd"]]
    )
    return {
        name: (float(mean), float(sd))
        for name, mean, sd in zip(BACKGROUND_PARAMETERS, means, sds, strict=True)
    }


def _check_comparable(data_a: Mapping[str, Any], data_b: Mapping[str, Any]) -> None:
    """Raise unless both fits use the same spectrum and shared priors."""
    for key in _SHARED_DATA_KEYS:
        if not np.array_equal(np.asarray(data_a[key]), np.asarray(data_b[key])):
            msg = f"The two fits use different '{key}'; compare the same spectrum."
            raise ValueError(msg)
    if fixed_parameters(data_a) == fixed_parameters(data_b):
        msg = "The two fits fix the same parameters; there is nothing to compare."
        raise ValueError(msg)
    priors_a, priors_b = _prior_table(data_a), _prior_table(data_b)
    shared = set(free_parameters(data_a)) & set(free_parameters(data_b))
    differing = sorted(name for name in shared if priors_a[name] != priors_b[name])
    if differing:
        msg = f"Shared parameters have different priors: {differing}."
        raise ValueError(msg)


@dataclass(frozen=True)
class BayesFactorResult:
    """Bayes factor ``BF_ab = Z_a / Z_b`` between two background variants."""

    model_a: str
    model_b: str
    evidence_a: BridgeSamplingResult
    evidence_b: BridgeSamplingResult

    @property
    def log_bf(self) -> float:
        """``ln BF_ab``; positive values favour ``model_a``."""
        return self.evidence_a.log_ml - self.evidence_b.log_ml

    @property
    def log_bf_sd(self) -> float:
        """Bridge-sampling uncertainty of ``log_bf`` (errors in quadrature)."""
        return float(np.hypot(self.evidence_a.log_ml_sd, self.evidence_b.log_ml_sd))

    @property
    def log10_bf(self) -> float:
        return self.log_bf / np.log(10.0)

    @property
    def probability_a(self) -> float:
        """Posterior probability of ``model_a`` for equal prior odds."""
        return float(expit(self.log_bf))

    @property
    def interpretation(self) -> str:
        """Kass & Raftery (1995) strength, applied to ``2 ln BF``."""
        favoured = self.model_a if self.log_bf >= 0 else self.model_b
        strength = next(
            label for bound, label in _KASS_RAFTERY if 2 * abs(self.log_bf) < bound
        )
        return f"{strength} evidence for '{favoured}'"

    def as_dict(self) -> dict[str, Any]:
        """Plain-type record, e.g. for ``write_yaml``."""
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "log_bf": float(self.log_bf),
            "log_bf_sd": self.log_bf_sd,
            "log10_bf": float(self.log10_bf),
            "probability_a": self.probability_a,
            "interpretation": self.interpretation,
            "evidence_a": asdict(self.evidence_a),
            "evidence_b": asdict(self.evidence_b),
        }


def compare_background_models(
    fit_a: Any,
    data_a: Mapping[str, Any],
    fit_b: Any,
    data_b: Mapping[str, Any],
    *,
    seed: int = 0,
) -> BayesFactorResult:
    """Bayes factor between two variants of the background fit.

    Both fits must use the same spectrum (edges, counts, ``use_bin``) and
    fixed inputs, and the parameters free in both must have identical priors,
    so the comparison only concerns the fixed parameters. Typical use is the
    full model (``a``) against ``build_stan_data(..., fixed=NO_LINEAR)``
    (``b``). Each fit is a ``cmdstanpy.CmdStanMCMC`` or a reloaded
    ``posterior.nc``.

    The result depends on the priors of the parameters that differ, here the
    lognormal widths of ``c0`` and ``cQ``; check how it changes with them.
    """
    _check_comparable(data_a, data_b)
    return BayesFactorResult(
        model_a=model_label(data_a),
        model_b=model_label(data_b),
        evidence_a=background_log_marginal_likelihood(fit_a, data_a, seed=seed),
        evidence_b=background_log_marginal_likelihood(fit_b, data_b, seed=seed),
    )


def as_inference_data(posterior: Any, stan_data: Mapping[str, Any]) -> Any:
    """ArviZ data of a fit: converted from CmdStan, or returned unchanged."""
    if _is_cmdstan_fit(posterior):
        return to_inference_data(posterior, stan_data)
    return posterior


def pointwise_loo(
    posterior: Any, stan_data: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """PSIS-LOO ``elpd_i`` and Pareto ``k`` of each fitted bin.

    Both arrays follow the fitted bins in energy order
    (``np.flatnonzero(stan_data["use_bin"])``). ``posterior`` is a
    ``CmdStanMCMC`` or a reloaded ``posterior.nc``.
    """
    import arviz as az  # noqa: PLC0415

    result = az.loo(as_inference_data(posterior, stan_data), pointwise=True)
    return (
        np.asarray(result.elpd_i.values, dtype=float).ravel(),
        np.asarray(result.pareto_k.values, dtype=float).ravel(),
    )


def loo_compare_background_models(
    fit_a: Any,
    data_a: Mapping[str, Any],
    fit_b: Any,
    data_b: Mapping[str, Any],
) -> pd.DataFrame:
    """PSIS-LOO comparison of two background variants with ``arviz.compare``.

    Each fitted bin is one observation, so this is leave-one-bin-out over the
    bins with ``use_bin = 1``. The table gives each model's ELPD, the
    difference to the best model (``elpd_diff``, negative for the worse
    model) and its standard error (``dse``), the probability that a model is
    worse than the best (``p_worse``), and diagnostic messages
    (``diag_diff``, ``diag_elpd``). Values are not rounded. Unlike the Bayes
    factor, it depends only weakly on the priors. Each fit is a
    ``CmdStanMCMC`` or a reloaded ``posterior.nc``.
    """
    import arviz as az  # noqa: PLC0415

    _check_comparable(data_a, data_b)
    return az.compare(
        {
            model_label(data_a): as_inference_data(fit_a, data_a),
            model_label(data_b): as_inference_data(fit_b, data_b),
        },
        round_to="none",
    )

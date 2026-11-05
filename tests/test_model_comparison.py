# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

import shutil

import numpy as np
import pytest
from scipy.special import gammaln
from scipy.stats import gamma, multivariate_normal

from bosonic_dm.fitting import background_shapes as shapes
from bosonic_dm.fitting.background_fit import (
    NO_LINEAR,
    STAN_FILE,
    background_log_density,
    build_stan_data,
    free_parameters,
    inits_from_prefit,
    priors_from_prefit,
    run_background_fit,
)
from bosonic_dm.fitting.model_comparison import (
    background_log_marginal_likelihood,
    bridge_sampling,
    compare_background_models,
    loo_compare_background_models,
    model_label,
)

Q = 2039.0
PREFIT = {
    "N_2nu": 20000.0,
    "N_ar39": 40000.0,
    "linear_parameters": [20.0, 20.0],
    "ar39_parameters": [1.2, 1.5, 1.0, 2.0, 3.0],
}


def test_bridge_sampling_recovers_gaussian_normalization() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=(3, 3))
    dist = multivariate_normal([1.0, -2.0, 0.5], a @ a.T + 0.5 * np.eye(3))
    log_z = 3.7
    draws = dist.rvs(size=4 * 1000, random_state=rng).reshape(4, 1000, 3)

    result = bridge_sampling(lambda x: log_z + dist.logpdf(x), draws, seed=2)

    assert result.converged
    assert result.log_ml_sd < 0.01
    assert abs(result.log_ml - log_z) < max(4 * result.log_ml_sd, 2e-3)


def test_bridge_sampling_recovers_poisson_gamma_evidence() -> None:
    """A positive parameter sampled on the log scale, Jacobian included."""
    rng = np.random.default_rng(4)
    alpha, beta = 3.0, 0.5
    counts = rng.poisson(7.0, size=20)
    n, total = len(counts), counts.sum()
    exact = (
        alpha * np.log(beta)
        - gammaln(alpha)
        + gammaln(alpha + total)
        - (alpha + total) * np.log(beta + n)
        - gammaln(counts + 1).sum()
    )
    posterior = gamma(alpha + total, scale=1 / (beta + n))
    draws = np.log(posterior.rvs(size=(4, 1000), random_state=rng))[..., None]

    def log_density(u: np.ndarray) -> np.ndarray:
        rate = np.exp(u[:, 0])
        log_prior = gamma(alpha, scale=1 / beta).logpdf(rate) + u[:, 0]
        log_lik = (
            counts * np.log(rate[:, None]) - rate[:, None] - gammaln(counts + 1)
        ).sum(axis=1)
        return log_prior + log_lik

    result = bridge_sampling(log_density, draws, seed=3)

    assert abs(result.log_ml - exact) < max(4 * result.log_ml_sd, 2e-3)


@pytest.mark.parametrize("shape", [(100, 3), (2, 3, 2)])
def test_bridge_sampling_rejects_bad_draws(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        bridge_sampling(lambda x: np.zeros(len(x)), np.zeros(shape))


def test_model_labels() -> None:
    spectrum = _spectrum(np.zeros(4, dtype=int), np.arange(60.0, 141.0, 20.0))
    priors = priors_from_prefit(PREFIT)
    assert model_label(build_stan_data(spectrum, priors)) == "full"
    assert (
        model_label(build_stan_data(spectrum, priors, fixed=NO_LINEAR)) == "no linear"
    )
    assert model_label(build_stan_data(spectrum, priors, fixed={"cQ": 0.5})) == "cQ=0.5"


def _spectrum(counts: np.ndarray, edges: np.ndarray) -> dict:
    return {
        "bin_edges": edges,
        "counts_analysis": counts,
        "fit_mask": np.ones(len(counts), dtype=bool),
    }


def _toy_spectrum() -> dict:
    """Primakoff-Rosen + Ar39 + a large flat continuum (20 counts/keV)."""
    edges = np.arange(60.0, 1101.0, 20.0)
    expected = (
        PREFIT["N_2nu"] * np.diff(shapes.primakoff_rosen_cdf(edges, Q))
        + PREFIT["N_ar39"]
        * np.diff(
            shapes.modified_beta_g_cdf(edges, *PREFIT["ar39_parameters"], 0.0, 565.8)
        )
        + 20.0 * np.diff(edges)
    )
    counts = np.random.default_rng(7).poisson(expected)
    spectrum = _spectrum(counts, edges)
    spectrum["fit_mask"][(edges[:-1] > 570) & (edges[:-1] < 600)] = False
    return spectrum


@pytest.fixture(scope="module")
def background_model(tmp_path_factory: pytest.TempPathFactory):
    cmdstanpy = pytest.importorskip("cmdstanpy")
    try:
        cmdstanpy.cmdstan_path()
    except ValueError:
        pytest.skip("CmdStan is not installed")
    target = tmp_path_factory.mktemp("stan_model_comparison") / STAN_FILE.name
    shutil.copyfile(STAN_FILE, target)
    return cmdstanpy.CmdStanModel(stan_file=str(target))


@pytest.mark.parametrize("fixed", [None, NO_LINEAR, {"c0": 4.0}])
def test_numpy_log_density_matches_stan(background_model, fixed: dict | None) -> None:
    data = build_stan_data(_toy_spectrum(), priors_from_prefit(PREFIT), fixed=fixed)
    point = {
        "N_2nu": 19000.0,
        "N_ar39": 41000.0,
        "c0": 18.0,
        "cQ": 22.0,
        "mu": 1.1,
        "sigma": 1.6,
        "tau": 0.9,
        "base_alpha": 2.1,
        "base_beta": 2.9,
    }
    names = free_parameters(data)
    params = {
        name: value
        for name, value in point.items()
        if name in names and name not in ("c0", "cQ")
    }
    for name in ("c0", "cQ"):
        params[f"{name}_free"] = [point[name]] if name in names else []

    stan_lp = background_model.log_prob(params, data, jacobian=True, sig_figs=16)
    numpy_lp = background_log_density(np.log([[point[n] for n in names]]), data)

    assert numpy_lp[0] == pytest.approx(stan_lp["lp__"].iloc[0], rel=1e-10)


@pytest.fixture(scope="module")
def toy_fits(background_model, tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Short fits of the full and no-linear variants to the same spectrum."""
    spectrum = _toy_spectrum()
    priors = priors_from_prefit(PREFIT)
    output = tmp_path_factory.mktemp("comparison_fits")
    fits = {}
    for label, fixed in (("full", None), ("no_linear", NO_LINEAR)):
        data = build_stan_data(spectrum, priors, fixed=fixed)
        fit = run_background_fit(
            background_model,
            data,
            seed=13,
            chains=2,
            iter_warmup=300,
            iter_sampling=300,
            output_dir=output / label,
            inits=inits_from_prefit(PREFIT, chains=2, fixed=fixed),
        )
        fits[label] = (fit, data)
    return fits


def test_bayes_factor_favours_the_linear_term(toy_fits: dict) -> None:
    result = compare_background_models(*toy_fits["full"], *toy_fits["no_linear"])

    assert (result.model_a, result.model_b) == ("full", "no linear")
    assert result.evidence_a.converged
    assert result.evidence_b.converged
    assert result.log_bf > 10
    assert result.log_bf > 5 * result.log_bf_sd
    assert result.probability_a > 0.99
    assert result.interpretation == "very strong evidence for 'full'"
    record = result.as_dict()
    assert record["log_bf"] == pytest.approx(result.log_bf)
    assert record["evidence_b"]["n_proposal"] == result.evidence_b.n_proposal


# Short toy fits can have a few bins with high Pareto k; that warning is real
# diagnostic output, not a failure of the helper.
@pytest.mark.filterwarnings("ignore:Estimated shape parameter of Pareto")
def test_loo_comparison_has_both_models(toy_fits: dict) -> None:
    table = loo_compare_background_models(*toy_fits["full"], *toy_fits["no_linear"])

    assert set(table.index) == {"full", "no linear"}
    assert table.loc["full", "elpd"] > table.loc["no linear", "elpd"]


def test_mismatched_inputs_are_rejected(toy_fits: dict) -> None:
    fit_full, data_full = toy_fits["full"]
    fit_nl, data_nl = toy_fits["no_linear"]

    with pytest.raises(ValueError, match="same parameters"):
        compare_background_models(fit_full, data_full, fit_full, data_full)
    other_counts = {**data_nl, "counts": data_nl["counts"] + 1}
    with pytest.raises(ValueError, match="'counts'"):
        compare_background_models(fit_full, data_full, fit_nl, other_counts)
    wider = {**data_nl, "log_shape_prior_sd": data_nl["log_shape_prior_sd"] * 2}
    with pytest.raises(ValueError, match="different priors"):
        compare_background_models(fit_full, data_full, fit_nl, wider)
    # Data that differ from the fitted data fail the lp__ consistency check.
    with pytest.raises(ValueError, match="lp__"):
        background_log_marginal_likelihood(fit_nl, wider)

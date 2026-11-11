# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

import arviz as az
import numpy as np
import pytest
import xarray as xr
from scipy.special import gammaln
from scipy.stats import gamma, multivariate_normal

from bosonic_dm.fitting.background_fit import (
    NO_LINEAR,
    background_log_density,
    build_stan_data,
    free_parameters,
    load_stan_data,
    priors_from_prefit,
    to_inference_data,
)
from bosonic_dm.fitting.model_comparison import (
    background_log_marginal_likelihood,
    bridge_sampling,
    bridge_sampling_estimate,
    bridge_sampling_terms,
    check_bridge_sampling,
    compare_background_models,
    loo_compare_background_models,
    model_label,
    pointwise_loo,
)


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


def test_bridge_sampling_terms_reproduce_the_estimate() -> None:
    rng = np.random.default_rng(5)
    dist = multivariate_normal([0.5, -1.0], [[1.0, 0.3], [0.3, 0.5]])
    draws = dist.rvs(size=2 * 400, random_state=rng).reshape(2, 400, 2)

    def log_density(x: np.ndarray) -> np.ndarray:
        return 1.5 + dist.logpdf(x)

    terms = bridge_sampling_terms(log_density, draws, seed=4)
    reference = bridge_sampling(log_density, draws, seed=4)

    assert bridge_sampling_estimate(terms) == reference
    assert terms.posterior_draws.shape == (2, 200, 2)
    reused = bridge_sampling_terms(
        log_density, draws, seed=4, log_q_posterior=terms.log_q_posterior
    )
    assert bridge_sampling_estimate(reused) == reference
    with pytest.raises(ValueError, match="log_q_posterior"):
        bridge_sampling_terms(log_density, draws, log_q_posterior=np.zeros(3))


def test_model_labels(toy_prefit: dict) -> None:
    spectrum = _spectrum(np.zeros(4, dtype=int), np.arange(60.0, 141.0, 20.0))
    priors = priors_from_prefit(toy_prefit)
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


@pytest.mark.parametrize("fixed", [None, NO_LINEAR, {"c0": 4.0}])
def test_numpy_log_density_matches_stan(
    background_model, toy_spectrum: dict, toy_prefit: dict, fixed: dict | None
) -> None:
    data = build_stan_data(toy_spectrum, priors_from_prefit(toy_prefit), fixed=fixed)
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
    # Not rounded: equal to the ELPD of a direct PSIS-LOO computation.
    fit, data = toy_fits["full"]
    exact = az.loo(to_inference_data(fit, data)).elpd
    assert table.loc["full", "elpd"] == pytest.approx(exact, rel=1e-12)

    elpd_i, pareto_k = pointwise_loo(fit, data)
    assert elpd_i.shape == pareto_k.shape == (int(data["use_bin"].sum()),)
    assert elpd_i.sum() == pytest.approx(exact, rel=1e-10)


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


def test_load_stan_data_round_trip(toy_fits: dict, saved_toy_fits: dict) -> None:
    for label, (_, data) in toy_fits.items():
        loaded = load_stan_data(saved_toy_fits[label])

        assert set(loaded) == set(data)
        for key, value in data.items():
            np.testing.assert_array_equal(loaded[key], value, err_msg=key)


def test_evidence_from_saved_posterior_matches_fit(
    toy_fits: dict, saved_toy_fits: dict
) -> None:
    fit, data = toy_fits["no_linear"]
    tree = xr.load_datatree(saved_toy_fits["no_linear"] / "posterior.nc")

    from_fit = background_log_marginal_likelihood(fit, data)
    from_disk = background_log_marginal_likelihood(
        tree, load_stan_data(saved_toy_fits["no_linear"])
    )

    assert from_disk.log_ml == pytest.approx(from_fit.log_ml, rel=1e-12)
    assert from_disk.log_ml_sd == pytest.approx(from_fit.log_ml_sd, rel=1e-9)


def test_check_bridge_sampling(toy_fits: dict) -> None:
    fit, data = toy_fits["full"]

    check = check_bridge_sampling(fit, data, seeds=(1, 2, 3))

    assert check.parameters == free_parameters(data)
    assert check.result == background_log_marginal_likelihood(fit, data)
    assert check.seeds == (1, 2, 3)
    assert len(check.by_seed) == 3
    assert check.n_draws == (37, 75, 150, 300)
    assert check.by_draws[-1] == check.result
    values = [r.log_ml for r in check.by_seed]
    assert np.all(np.isfinite(values))
    assert np.ptp(values) < 1.0

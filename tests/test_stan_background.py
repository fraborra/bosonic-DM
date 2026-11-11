# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

import itertools
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import beta as beta_dist
from scipy.stats import poisson

cmdstanpy = pytest.importorskip("cmdstanpy")


@pytest.fixture(scope="module")
def background_model(
    tmp_path_factory: pytest.TempPathFactory,
) -> cmdstanpy.CmdStanModel:
    """Compile in a temporary directory, leaving the source tree clean."""
    try:
        cmdstanpy.cmdstan_path()
    except ValueError:
        pytest.skip("CmdStan is not installed")
    source = Path(__file__).resolve().parents[1] / "stan" / "background.stan"
    target = tmp_path_factory.mktemp("stan_background") / source.name
    shutil.copyfile(source, target)
    return cmdstanpy.CmdStanModel(stan_file=str(target))


def _linear_switches(fixed: dict[str, float]) -> dict[str, float]:
    """Stan data entries that fix ``c0`` and/or ``cQ`` at the given values."""
    return {
        **{f"fit_{name}": int(name not in fixed) for name in ("c0", "cQ")},
        **{f"{name}_fixed": fixed.get(name, 0.0) for name in ("c0", "cQ")},
    }


# Full model, model without the linear term, and c0 fixed at a nonzero value.
VARIANTS = [{}, {"c0": 0.0, "cQ": 0.0}, {"c0": 0.45}]


@pytest.mark.parametrize("fixed", VARIANTS)
@pytest.mark.parametrize(
    ("edges", "shape"),
    [
        ([0, 1, 50, 300, 565.8, 1000, 2000, 2039], [1.7, 2.3, 0.6]),
        ([100, 150, 220, 600, 1400], [0.8, 1.2, 2.5]),
        ([0, 2039], [1.0, 1.0, 1.0]),
    ],
)
def test_bin_integrals(
    background_model: cmdstanpy.CmdStanModel,
    tmp_path: Path,
    edges: list[float],
    shape: list[float],
    fixed: dict[str, float],
) -> None:
    """Stan component integrals match the supplied Python shapes."""
    q_value = 2039.0
    n_2nu, n_ar39 = 1200.0, 750.0
    c0, c_q = fixed.get("c0", 0.3), fixed.get("cQ", 0.7)
    base_alpha, base_beta, loc, scale = 2.1, 3.2, 5.0, 560.8
    mu, sigma, tau = shape
    bin_edges = np.asarray(edges, dtype=float)
    data = {
        "n_bins": len(edges) - 1,
        "edges": edges,
        "counts": [10] * (len(edges) - 1),
        "use_bin": [1] * (len(edges) - 1),
        "Q": q_value,
        "loc": loc,
        "scale": scale,
        **_linear_switches(fixed),
        "log_rate_prior_mean": np.log([n_2nu, n_ar39, 0.3, 0.7]),
        "log_rate_prior_sd": [1.0] * 4,
        "log_shape_prior_mean": np.log([*shape, base_alpha, base_beta]),
        "log_shape_prior_sd": [0.5] * 5,
    }
    inits = {
        "N_2nu": n_2nu,
        "N_ar39": n_ar39,
        "c0_free": [] if "c0" in fixed else [c0],
        "cQ_free": [] if "cQ" in fixed else [c_q],
        "mu": mu,
        "sigma": sigma,
        "tau": tau,
        "base_alpha": base_alpha,
        "base_beta": base_beta,
    }
    background_model.diagnose(data=data, inits=inits, epsilon=1e-5)
    # Fixed coefficients leave the sampled space: one gradient per free parameter.
    gradient = background_model.log_prob(inits, data).columns.drop("lp__")
    assert len(gradient) == 9 - len(fixed)
    fit = background_model.sample(
        data=data,
        inits=inits,
        fixed_param=True,
        chains=1,
        iter_sampling=1,
        seed=8123,
        sig_figs=16,
        output_dir=str(tmp_path),
    )

    def primakoff_rosen(energy: float) -> float:
        m_e = 510.99895
        q = q_value / m_e
        t = energy / m_e
        poly = t**4 + 10 * t**3 + 40 * t**2 + 60 * t + 30
        norm = q**7 * (1980 + 990 * q + 220 * q**2 + 22 * q**3 + q**4) / 2772
        return n_2nu * t * poly * (q - t) ** 5 / (m_e * norm)

    expected_pr = np.array(
        [quad(primakoff_rosen, lo, hi)[0] for lo, hi in itertools.pairwise(edges)]
    )
    midpoint = (bin_edges[1:] + bin_edges[:-1]) / (2 * q_value)
    expected_linear = np.diff(bin_edges) * (c0 * (1 - midpoint) + c_q * midpoint)
    u = beta_dist.cdf(bin_edges, base_alpha, base_beta, loc=loc, scale=scale)
    r = tau * u / (1 - u + tau * u)
    expected_ar = n_ar39 * np.diff(beta_dist.cdf(r, mu, sigma))
    assert fit.stan_variable("c0")[0] == pytest.approx(c0)
    assert fit.stan_variable("cQ")[0] == pytest.approx(c_q)
    for name, expected in (
        ("expected_2nu", expected_pr),
        ("expected_linear", expected_linear),
        ("expected_ar39", expected_ar),
        ("expected_counts", expected_pr + expected_linear + expected_ar),
    ):
        np.testing.assert_allclose(
            fit.stan_variable(name)[0], expected, rtol=1e-9, atol=1e-9
        )


def _masked_data(use_bin: list[int], counts: list[int]) -> dict:
    edges = [60.0, 100.0, 200.0, 300.0, 400.0, 565.8, 800.0, 1100.0]
    return {
        "n_bins": len(edges) - 1,
        "edges": edges,
        "counts": counts,
        "use_bin": use_bin,
        "Q": 2039.0,
        "loc": 0.0,
        "scale": 565.8,
        **_linear_switches({}),
        "log_rate_prior_mean": np.log([1200.0, 750.0, 0.3, 0.7]),
        "log_rate_prior_sd": [1.0] * 4,
        "log_shape_prior_mean": np.log([1.7, 2.3, 0.6, 2.1, 3.2]),
        "log_shape_prior_sd": [0.5] * 5,
    }


_POINT = {
    "N_2nu": 1200.0,
    "N_ar39": 750.0,
    "c0_free": [0.3],
    "cQ_free": [0.7],
    "mu": 1.7,
    "sigma": 2.3,
    "tau": 0.6,
    "base_alpha": 2.1,
    "base_beta": 3.2,
}


def test_masked_bins_do_not_enter_likelihood(
    background_model: cmdstanpy.CmdStanModel, tmp_path: Path
) -> None:
    """Masked bins drop exactly their Poisson terms; their counts are ignored."""
    counts = [40, 90, 120, 80, 60, 30, 25]
    use_bin = [1, 1, 0, 1, 0, 1, 1]

    def log_prob(data: dict) -> float:
        return float(background_model.log_prob(_POINT, data)["lp__"].iloc[0])

    lp_all = log_prob(_masked_data([1] * 7, counts))
    lp_masked = log_prob(_masked_data(use_bin, counts))
    # Peak counts inside masked bins must not change the posterior density.
    peaked = counts.copy()
    peaked[2] += 5000
    peaked[4] += 3000
    assert log_prob(_masked_data(use_bin, peaked)) == pytest.approx(lp_masked)

    fit = background_model.sample(
        data=_masked_data(use_bin, counts),
        inits=_POINT,
        fixed_param=True,
        chains=1,
        iter_sampling=1,
        seed=1,
        sig_figs=16,
        output_dir=str(tmp_path),
    )
    expected = fit.stan_variable("expected_counts")[0]
    masked = np.flatnonzero(np.asarray(use_bin) == 0)
    # lp__ keeps every normalizing constant, including -log(n!).
    dropped = sum(poisson.logpmf(counts[b], expected[b]) for b in masked)
    assert lp_all - lp_masked == pytest.approx(dropped, rel=1e-8)

    # log_lik covers fitted bins only; predictions cover every bin.
    assert fit.stan_variable("log_lik").shape == (1, sum(use_bin))
    assert fit.stan_variable("counts_rep").shape == (1, len(counts))


def test_all_masked_bins_are_rejected(background_model: cmdstanpy.CmdStanModel) -> None:
    with pytest.raises(RuntimeError):
        background_model.log_prob(_POINT, _masked_data([0] * 7, [1] * 7))


def test_subnormal_ar39_cdf_argument_keeps_gradient_finite(
    background_model: cmdstanpy.CmdStanModel,
) -> None:
    """A subnormal argument of the outer Ar-39 beta CDF must not crash the gradient.

    With ``base_alpha = 100`` the base CDF at ``loc + 1e-3 scale`` is
    ``u ~ 1e-296``, and ``tau = 1e-14`` maps it to ``r ~ 1e-310``. For
    ``mu < 1`` Boost's ``ibeta_derivative`` overflows at such ``r`` and throws
    an error that Stan cannot recover from, which stops the chain.
    """
    point = {**_POINT, "mu": 0.8, "tau": 1e-14, "base_alpha": 100.0}
    data = {
        **_masked_data([1] * 7, [10] * 7),
        "edges": [0.0, 0.5658, 100.0, 200.0, 300.0, 565.8, 800.0, 1100.0],
    }
    r = point["tau"] * beta_dist.cdf(1e-3, point["base_alpha"], point["base_beta"])
    assert 0 < r < np.finfo(float).tiny
    log_prob = background_model.log_prob(point, data)
    assert np.isfinite(log_prob.to_numpy()).all()

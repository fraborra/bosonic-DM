# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import poisson

from bosonic_dm.fitting.signal_fit import (
    build_signal_stan_data,
    run_signal_fit,
    save_signal_fit,
    suggest_rate_upper,
    summarize_signal_fit,
)
from bosonic_dm.fitting.signal_template import (
    GroupSignalInputs,
    SignalWindow,
    gaussian_bin_probabilities,
)
from bosonic_dm.yaml_io import read_yaml

cmdstanpy = pytest.importorskip("cmdstanpy")


@pytest.fixture(scope="module")
def signal_model(tmp_path_factory: pytest.TempPathFactory) -> cmdstanpy.CmdStanModel:
    """Compile in a temporary directory, leaving the source tree clean."""
    try:
        cmdstanpy.cmdstan_path()
    except ValueError:
        pytest.skip("CmdStan is not installed")
    source = Path(__file__).resolve().parents[1] / "stan" / "signal.stan"
    target = tmp_path_factory.mktemp("stan_signal") / source.name
    shutil.copyfile(source, target)
    return cmdstanpy.CmdStanModel(stan_file=str(target))


EDGES = np.arange(488.0, 513.0)
MASS = 500.0


def _window(
    counts: np.ndarray | None = None,
    use_bin: np.ndarray | None = None,
    b_levels: tuple[float, ...] = (5.0, 2.0),
    exposures: tuple[float, ...] = (10.0, 4.0),
    sigmas: tuple[float, ...] = (1.0, 1.4),
    efficiency: float = 0.9,
) -> SignalWindow:
    n_bins = len(EDGES) - 1
    groups = {}
    for g, (level, exposure, sigma) in enumerate(
        zip(b_levels, exposures, sigmas, strict=True)
    ):
        probabilities = gaussian_bin_probabilities(EDGES, MASS, sigma)
        groups[f"g{g}"] = GroupSignalInputs(
            counts=(
                np.full(n_bins, int(level), dtype=np.int64)
                if counts is None
                else np.asarray(counts[g], dtype=np.int64)
            ),
            b_hat=np.full(n_bins, level),
            signal_template=exposure * efficiency * probabilities,
            use_bin=(
                np.ones(n_bins, dtype=bool)
                if use_bin is None
                else np.asarray(use_bin[g], dtype=bool)
            ),
            exposure_kg_yr=exposure,
            efficiency=efficiency,
            efficiency_unc=0.01,
            efficiency_exposure_kg_yr=exposure,
            fwhm_keV=sigma / 0.4246609,
            fwhm_unc_keV=0.0,
            sigma_keV=sigma,
            centroid_keV=MASS,
            centroid_offset_keV=0.0,
            centroid_offset_unc_keV=0.0,
            centroid_offset_extrapolated=False,
            window_containment=float(probabilities.sum()),
        )
    return SignalWindow(
        mass_keV=MASS,
        half_width_keV=12.5,
        bin_edges=EDGES.copy(),
        groups=groups,
        background_sha256={"g0": {"bhat.parquet": "abc"}},
    )


def _grid_upper_limit(data: dict, credibility: float = 0.9) -> float:
    """Upper limit of the uniform-prior posterior on a fine rate grid."""
    used = np.asarray(data["use_bin"], dtype=bool)
    counts = np.asarray(data["counts"])[used]
    b_hat = np.asarray(data["b_hat"])[used]
    template = np.asarray(data["signal_template"])[used]
    rates = np.linspace(0.0, data["rate_upper"], 200_001)
    log_like = poisson.logpmf(counts, b_hat + rates[:, None] * template).sum(axis=1)
    density = np.exp(log_like - log_like.max())
    cdf = np.cumsum(density)
    cdf /= cdf[-1]
    return float(np.interp(credibility, cdf, rates))


def test_log_density_matches_numpy(signal_model: cmdstanpy.CmdStanModel) -> None:
    counts = np.random.default_rng(3).poisson(6.0, size=(2, len(EDGES) - 1))
    use_bin = np.ones_like(counts, dtype=bool)
    use_bin[0, 5:8] = False
    data = build_signal_stan_data(_window(counts, use_bin), rate_upper=50.0)

    def numpy_log_like(rate: float) -> float:
        lam = np.asarray(data["b_hat"]) + rate * np.asarray(data["signal_template"])
        return float(poisson.logpmf(counts, lam)[use_bin].sum())

    def stan_lp(rate: float) -> float:
        frame = signal_model.log_prob({"R": rate}, data, jacobian=False, sig_figs=16)
        return float(frame["lp__"].iloc[0])

    # Sampling statements drop -log(n!), so compare differences in R.
    assert stan_lp(3.0) - stan_lp(0.5) == pytest.approx(
        numpy_log_like(3.0) - numpy_log_like(0.5), rel=1e-10
    )


def test_masked_bins_do_not_enter_likelihood(
    signal_model: cmdstanpy.CmdStanModel,
) -> None:
    counts = np.full((2, len(EDGES) - 1), 5)
    use_bin = np.ones_like(counts, dtype=bool)
    use_bin[0, 10:14] = False
    peaked = counts.copy()
    peaked[0, 10:14] += 1000

    def lp(values: np.ndarray) -> float:
        data = build_signal_stan_data(_window(values, use_bin), rate_upper=50.0)
        return float(signal_model.log_prob({"R": 2.0}, data)["lp__"].iloc[0])

    assert lp(peaked) == pytest.approx(lp(counts))


def test_all_masked_bins_are_rejected(signal_model: cmdstanpy.CmdStanModel) -> None:
    use_bin = np.zeros((2, len(EDGES) - 1), dtype=bool)
    data = build_signal_stan_data(_window(use_bin=use_bin), rate_upper=50.0)
    with pytest.raises(RuntimeError):
        signal_model.log_prob({"R": 1.0}, data)


def test_upper_limit_matches_grid_posterior(
    signal_model: cmdstanpy.CmdStanModel, tmp_path: Path
) -> None:
    window = _window()
    rng = np.random.default_rng(11)
    counts = rng.poisson(window.stacked("b_hat"))
    window = _window(counts)
    data = build_signal_stan_data(window, rate_upper=suggest_rate_upper(window))

    fit = run_signal_fit(signal_model, data, seed=7, output_dir=tmp_path)
    summary = summarize_signal_fit(fit, data)

    assert summary.passed, summary.diagnostics
    expected = _grid_upper_limit(data)
    tolerance = 4 * summary.rate["upper_limit_mcse"]
    assert abs(summary.upper_limit - expected) < tolerance
    assert summary.diagnostics["upper_limit_to_prior_bound"] < 0.1


def test_injected_signal_is_recovered(
    signal_model: cmdstanpy.CmdStanModel, tmp_path: Path
) -> None:
    true_rate = 20.0
    nominal = _window()
    asimov = np.rint(
        nominal.stacked("b_hat") + true_rate * nominal.stacked("signal_template")
    )
    window = _window(asimov)
    data = build_signal_stan_data(window, rate_upper=suggest_rate_upper(window))

    fit = run_signal_fit(signal_model, data, seed=5, output_dir=tmp_path / "fit")
    summary = summarize_signal_fit(fit, data)

    assert summary.passed, summary.diagnostics
    assert abs(summary.rate["mean"] - true_rate) < 2 * summary.rate["sd"]
    assert summary.upper_limit > true_rate

    paths = save_signal_fit(
        fit, summary, window, data, tmp_path / "out", provenance={"test": True}
    )
    record = read_yaml(paths["summary"])
    assert record["mass_keV"] == MASS
    assert record["rate"]["upper_limit"] == pytest.approx(summary.upper_limit)
    assert record["groups"]["g0"]["n_fit_bins"] == len(EDGES) - 1
    assert record["background_fit_sha256"] == {"g0": {"bhat.parquet": "abc"}}
    assert "conditional" in record["interpretation"]
    assert np.load(paths["draws"]).shape == summary.rate_draws.shape

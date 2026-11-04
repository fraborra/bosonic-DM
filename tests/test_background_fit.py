# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from bosonic_dm.fitting.background_fit import (
    BACKGROUND_PARAMETERS,
    STAN_FILE,
    USABILITY_REMINDER,
    _excluded_intervals,
    build_stan_data,
    inits_from_prefit,
    load_background_fit,
    priors_from_prefit,
    run_background_fit,
    save_background_fit,
    summarize_background_fit,
)
from bosonic_dm.yaml_io import read_yaml

PREFIT = {
    "N_2nu": 1200.0,
    "N_ar39": 750.0,
    "linear_parameters": [0.3, 0.7],
    "ar39_parameters": [1.7, 2.3, 0.6, 2.1, 3.2],
}


def _spectrum(n_bins: int = 6, start: float = 60.0, width: float = 10.0) -> dict:
    edges = start + width * np.arange(n_bins + 1)
    mask = np.ones(n_bins, dtype=bool)
    mask[2:4] = False
    return {
        "bin_edges": edges,
        "counts_analysis": np.arange(n_bins) + 5,
        "fit_mask": mask,
    }


def test_priors_are_read_by_key_name() -> None:
    priors = priors_from_prefit(PREFIT, rate_sd=0.7, shape_sd=0.4)

    np.testing.assert_allclose(
        np.exp(priors["log_rate_prior_mean"]), [1200.0, 750.0, 0.3, 0.7]
    )
    np.testing.assert_allclose(
        np.exp(priors["log_shape_prior_mean"]), PREFIT["ar39_parameters"]
    )
    np.testing.assert_array_equal(priors["log_rate_prior_sd"], [0.7] * 4)
    np.testing.assert_array_equal(priors["log_shape_prior_sd"], [0.4] * 5)


@pytest.mark.parametrize(
    "override",
    [{"N_ar39": 0.0}, {"ar39_parameters": [1.0, 1.0]}, {"linear_parameters": [-1, 1]}],
)
def test_invalid_prefit_is_rejected(override: dict) -> None:
    with pytest.raises(ValueError):
        priors_from_prefit({**PREFIT, **override})


def test_inits_start_near_prefit_and_differ_between_chains() -> None:
    inits = inits_from_prefit(PREFIT, chains=3, jitter=0.1, seed=5)
    exact = inits_from_prefit(PREFIT, chains=2, jitter=0.0)

    assert len(inits) == 3
    assert set(inits[0]) == set(BACKGROUND_PARAMETERS)
    assert inits[0] != inits[1]
    assert inits == inits_from_prefit(PREFIT, chains=3, jitter=0.1, seed=5)
    assert exact[0] == exact[1]
    assert exact[0]["N_2nu"] == 1200.0
    assert exact[0]["N_ar39"] == 750.0
    assert exact[0]["cQ"] == 0.7
    assert exact[0]["base_beta"] == 3.2
    for init in inits:
        for name, value in exact[0].items():
            assert 0.5 * value < init[name] < 2.0 * value


@pytest.mark.parametrize("kwargs", [{"chains": 0}, {"jitter": -0.1}])
def test_invalid_inits_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        inits_from_prefit(PREFIT, **kwargs)


def test_per_chain_inits_must_match_chain_count() -> None:
    with pytest.raises(ValueError, match="per-chain inits"):
        run_background_fit(
            None, {}, seed=1, chains=4, inits=inits_from_prefit(PREFIT, chains=2)
        )


def test_stan_data_uses_fit_mask_and_keeps_real_counts() -> None:
    spectrum = _spectrum()
    data = build_stan_data(spectrum, priors_from_prefit(PREFIT))

    assert data["n_bins"] == 6
    np.testing.assert_array_equal(data["counts"], spectrum["counts_analysis"])
    np.testing.assert_array_equal(data["use_bin"], [1, 1, 0, 0, 1, 1])
    assert data["Q"] == 2039.0
    assert data["scale"] == 565.8


@pytest.mark.parametrize(
    "spectrum",
    [
        {**_spectrum(), "bin_edges": 1500.0 + 100.0 * np.arange(7)},
        {**_spectrum(), "fit_mask": np.zeros(6, dtype=bool)},
        {**_spectrum(), "counts_analysis": np.arange(5)},
        {**_spectrum(), "counts_analysis": np.full(6, 1.5)},
    ],
)
def test_invalid_stan_data_is_rejected(spectrum: dict) -> None:
    with pytest.raises(ValueError):
        build_stan_data(spectrum, priors_from_prefit(PREFIT))


def test_excluded_intervals_merge_adjacent_bins() -> None:
    edges = np.arange(8, dtype=float)
    used = np.array([1, 0, 0, 1, 0, 1, 1], dtype=bool)
    assert _excluded_intervals(edges, used) == [[1.0, 3.0], [4.0, 5.0]]


@pytest.fixture(scope="module")
def background_model(tmp_path_factory: pytest.TempPathFactory):
    cmdstanpy = pytest.importorskip("cmdstanpy")
    try:
        cmdstanpy.cmdstan_path()
    except ValueError:
        pytest.skip("CmdStan is not installed")
    target = tmp_path_factory.mktemp("stan_background_fit") / STAN_FILE.name
    shutil.copyfile(STAN_FILE, target)
    return cmdstanpy.CmdStanModel(stan_file=str(target))


def test_fit_summary_and_outputs(
    background_model, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A short fit yields B_hat on every bin and writes all products."""
    rng = np.random.default_rng(3)
    edges = np.arange(60.0, 1101.0, 20.0)
    centres = 0.5 * (edges[1:] + edges[:-1])
    truth = 40.0 * np.exp(-centres / 400.0) + 5.0
    mask = np.ones(len(centres), dtype=bool)
    mask[(centres > 570) & (centres < 600)] = False
    counts = rng.poisson(truth)
    counts[~mask] += 500  # A peak that must be ignored.
    spectrum = {"bin_edges": edges, "counts_analysis": counts, "fit_mask": mask}
    prefit = {
        "N_2nu": 2000.0,
        "N_ar39": 8000.0,
        "linear_parameters": [0.3, 0.3],
        "ar39_parameters": [1.0, 1.0, 1.0, 2.0, 3.0],
    }
    data = build_stan_data(spectrum, priors_from_prefit(prefit))

    fit = run_background_fit(
        background_model,
        data,
        seed=11,
        chains=2,
        iter_warmup=200,
        iter_sampling=200,
        output_dir=tmp_path / "csv",
        inits=inits_from_prefit(prefit, chains=2),
    )
    summary = summarize_background_fit(fit, data, n_saved_draws=50)

    assert USABILITY_REMINDER in caplog.text
    assert summary.b_hat.shape == (len(centres),)
    assert summary.expected_counts_draws.shape == (50, len(centres))
    np.testing.assert_array_equal(summary.used_in_fit, mask)
    # The masked peak does not pull B_hat up.
    assert np.all(summary.b_hat[~mask] < 100)
    np.testing.assert_allclose(
        sum(summary.component_medians.values()), summary.b_hat, rtol=0.2
    )

    paths = save_background_fit(
        fit, summary, data, tmp_path / "out", provenance={"group": "toy"}
    )
    assert all(path.exists() for path in paths.values())
    table = pl.read_parquet(paths["bhat"])
    np.testing.assert_allclose(table["B_hat"].to_numpy(), summary.b_hat)
    record = read_yaml(paths["summary"])
    assert record["provenance"] == {"group": "toy"}
    assert record["n_fit_bins"] == int(mask.sum())
    assert record["excluded_intervals_keV"] == [[580.0, 600.0]]
    assert record["sampler"]["seed"] == 11
    assert len(record["software"]["stan_file_sha256"]) == 64

    loaded = load_background_fit(tmp_path / "out")
    assert loaded.passed == summary.passed
    assert loaded.parameters == summary.parameters
    assert loaded.diagnostics == summary.diagnostics
    for name in (
        "bin_edges",
        "counts",
        "used_in_fit",
        "b_hat",
        "expected_counts_draws",
    ):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(summary, name))
    for name, values in summary.component_medians.items():
        np.testing.assert_array_equal(loaded.component_medians[name], values)

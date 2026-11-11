# Copyright (C) 2026 Francesco Borra
#

"""Shared toy fits of the full and no-linear background variants."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from bosonic_dm.fitting import background_shapes as shapes
from bosonic_dm.fitting.background_fit import (
    NO_LINEAR,
    STAN_FILE,
    build_stan_data,
    inits_from_prefit,
    priors_from_prefit,
    run_background_fit,
    save_background_fit,
    summarize_background_fit,
)

TOY_Q = 2039.0
TOY_PREFIT = {
    "N_2nu": 20000.0,
    "N_ar39": 40000.0,
    "linear_parameters": [20.0, 20.0],
    "ar39_parameters": [1.2, 1.5, 1.0, 2.0, 3.0],
}


def _toy_spectrum() -> dict:
    """Primakoff-Rosen + Ar39 + a large flat continuum (20 counts/keV)."""
    edges = np.arange(60.0, 1101.0, 20.0)
    expected = (
        TOY_PREFIT["N_2nu"] * np.diff(shapes.primakoff_rosen_cdf(edges, TOY_Q))
        + TOY_PREFIT["N_ar39"]
        * np.diff(
            shapes.modified_beta_g_cdf(
                edges, *TOY_PREFIT["ar39_parameters"], 0.0, 565.8
            )
        )
        + 20.0 * np.diff(edges)
    )
    counts = np.random.default_rng(7).poisson(expected)
    fit_mask = np.ones(len(counts), dtype=bool)
    fit_mask[(edges[:-1] > 570) & (edges[:-1] < 600)] = False
    return {"bin_edges": edges, "counts_analysis": counts, "fit_mask": fit_mask}


@pytest.fixture
def toy_prefit() -> dict:
    return {key: list(v) if isinstance(v, list) else v for key, v in TOY_PREFIT.items()}


@pytest.fixture
def toy_spectrum() -> dict:
    """Toy group spectrum with one window left out of the likelihood."""
    return _toy_spectrum()


@pytest.fixture(scope="session")
def background_model(tmp_path_factory: pytest.TempPathFactory):
    cmdstanpy = pytest.importorskip("cmdstanpy")
    try:
        cmdstanpy.cmdstan_path()
    except ValueError:
        pytest.skip("CmdStan is not installed")
    target = tmp_path_factory.mktemp("stan_model_comparison") / STAN_FILE.name
    shutil.copyfile(STAN_FILE, target)
    return cmdstanpy.CmdStanModel(stan_file=str(target))


@pytest.fixture(scope="session")
def toy_fits(background_model, tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Short fits of the full and no-linear variants to the same spectrum."""
    spectrum = _toy_spectrum()
    priors = priors_from_prefit(TOY_PREFIT)
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
            inits=inits_from_prefit(TOY_PREFIT, chains=2, fixed=fixed),
        )
        fits[label] = (fit, data)
    return fits


@pytest.fixture(scope="session")
def saved_toy_fits(toy_fits: dict, tmp_path_factory: pytest.TempPathFactory) -> dict:
    """``save_background_fit`` output directories of the toy fits."""
    root = tmp_path_factory.mktemp("saved_comparison_fits")
    paths: dict[str, Path] = {}
    for label, (fit, data) in toy_fits.items():
        summary = summarize_background_fit(fit, data)
        save_background_fit(fit, summary, data, root / label)
        paths[label] = root / label
    return paths

# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.integrate import quad

from bosonic_dm.fitting import background_shapes as shapes
from bosonic_dm.fitting.background_fit import (
    BACKGROUND_PARAMETERS,
    STAN_FILE,
    BackgroundFitSummary,
    build_stan_data,
    priors_from_prefit,
    run_background_fit,
    to_inference_data,
)
from bosonic_dm.plotting.background_fit import (
    excluded_window_excess,
    normalized_residuals,
    plot_background_chains,
    plot_background_corner,
    plot_background_fit_residuals,
    posterior_draws,
)

Q = 2039.0
AR39 = (1.7, 2.3, 0.6, 2.1, 3.2, 5.0, 560.8)  # mu, sigma, tau, alpha, beta, loc, scale


def test_primakoff_rosen_matches_its_cdf() -> None:
    assert quad(shapes.primakoff_rosen, 0, Q, args=(1200.0,))[0] == pytest.approx(
        1200.0
    )
    for energy in (100.0, 700.0, 1500.0):
        integral = quad(shapes.primakoff_rosen, 0, energy, args=(1.0,))[0]
        assert shapes.primakoff_rosen_cdf(energy) == pytest.approx(integral, rel=1e-10)
    np.testing.assert_array_equal(shapes.primakoff_rosen([-5.0, 2100.0], 1.0), 0.0)


def test_modified_beta_g_pdf_matches_its_cdf() -> None:
    assert quad(shapes.modified_beta_g_pdf, 5.0, 565.8, args=AR39)[0] == pytest.approx(
        1.0
    )
    for energy in (50.0, 300.0):
        integral = quad(shapes.modified_beta_g_pdf, 0, energy, args=AR39)[0]
        assert shapes.modified_beta_g_cdf(energy, *AR39) == pytest.approx(
            integral, rel=1e-6
        )
    # Off-support values are zero even when mu < 1 makes the density diverge at loc.
    np.testing.assert_array_equal(
        shapes.modified_beta_g_pdf([0.0, 600.0], 0.8, 1.2, 2.5, *AR39[3:]), 0.0
    )


def test_linear_background_endpoints() -> None:
    np.testing.assert_allclose(shapes.linear_background([0.0, Q], 0.3, 0.7), [0.3, 0.7])


def _summary(seed: int = 1) -> tuple[BackgroundFitSummary, np.ndarray]:
    """Model built from the Python shapes; a peak sits in one of two masked windows."""
    edges = np.arange(60.0, 1101.0, 1.0)
    expected = (
        5e5 * np.diff(shapes.primakoff_rosen_cdf(edges))
        + 2e6 * np.diff(shapes.modified_beta_g_cdf(edges, *AR39))
        + np.diff(edges)
        * shapes.linear_background(0.5 * (edges[1:] + edges[:-1]), 5, 2)
    )
    used = np.ones(len(expected), dtype=bool)
    used[(edges[:-1] >= 571) & (edges[:-1] < 595)] = False  # window with a peak
    used[(edges[:-1] >= 848) & (edges[:-1] < 872)] = False  # empty window
    centres = 0.5 * (edges[1:] + edges[:-1])
    peak = 3000 * np.exp(-0.5 * ((centres - 583.2) / 1.2) ** 2)
    rng = np.random.default_rng(seed)
    counts = rng.poisson(expected + peak)
    summary = BackgroundFitSummary(
        parameters={},
        diagnostics={},
        passed=True,
        bin_edges=edges,
        used_in_fit=used,
        b_hat=expected,
        component_medians={"expected_ar39": expected},
        expected_counts_draws=np.tile(expected, (5, 1)),
    )
    return summary, counts


def test_excluded_window_excess_flags_only_the_peak() -> None:
    summary, counts = _summary()
    table = excluded_window_excess(summary, counts)

    assert table["lo_keV"].to_list() == [571.0, 848.0]
    assert table["hi_keV"].to_list() == [595.0, 872.0]
    significance = table["significance"].to_numpy()
    assert significance[0] > 10
    assert abs(significance[1]) < 4


def test_residuals_are_standard_normal_without_peak() -> None:
    summary, counts = _summary()
    pulls = normalized_residuals(summary, counts)[summary.used_in_fit]
    assert abs(pulls.mean()) < 0.1
    assert pulls.std() == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("include_excluded", [False, True])
def test_residual_plot_is_saved(tmp_path: Path, include_excluded: bool) -> None:
    summary, counts = _summary()
    output = tmp_path / "residuals.png"
    fig = plot_background_fit_residuals(
        summary, counts, include_excluded=include_excluded, output_path=output
    )
    n_points = sum(
        len(line.get_xdata()) for line in fig.axes[1].lines[1:]
    )  # skip y=0 line
    plt.close(fig)

    assert output.stat().st_size > 0
    n_fit = int(summary.used_in_fit.sum())
    assert n_points == (len(counts) if include_excluded else n_fit)


@pytest.fixture(scope="module")
def short_fit(tmp_path_factory: pytest.TempPathFactory):
    cmdstanpy = pytest.importorskip("cmdstanpy")
    try:
        cmdstanpy.cmdstan_path()
    except ValueError:
        pytest.skip("CmdStan is not installed")
    workdir = tmp_path_factory.mktemp("stan_background_plots")
    shutil.copyfile(STAN_FILE, workdir / STAN_FILE.name)
    model = cmdstanpy.CmdStanModel(stan_file=str(workdir / STAN_FILE.name))
    edges = np.arange(60.0, 1101.0, 20.0)
    centres = 0.5 * (edges[1:] + edges[:-1])
    counts = np.random.default_rng(3).poisson(40.0 * np.exp(-centres / 400.0) + 5.0)
    spectrum = {
        "bin_edges": edges,
        "counts_analysis": counts,
        "fit_mask": np.ones(len(centres), dtype=bool),
    }
    prefit = {
        "N_2nu": 2000.0,
        "N_ar39": 8000.0,
        "linear_parameters": [0.3, 0.3],
        "ar39_parameters": [1.0, 1.0, 1.0, 2.0, 3.0],
    }
    data = build_stan_data(spectrum, priors_from_prefit(prefit))
    fit = run_background_fit(
        model,
        data,
        seed=11,
        chains=2,
        iter_warmup=100,
        iter_sampling=100,
        output_dir=workdir / "csv",
    )
    return fit, data


def test_posterior_draws_agree_between_cmdstan_and_arviz(short_fit) -> None:
    fit, data = short_fit
    from_fit = posterior_draws(fit)
    from_idata = posterior_draws(to_inference_data(fit, data))

    assert from_fit.shape == (2, 100, len(BACKGROUND_PARAMETERS))
    np.testing.assert_allclose(from_fit, from_idata)
    np.testing.assert_allclose(from_fit[..., 0].ravel(), fit.stan_variable("N_2nu"))


def test_chain_and_corner_plots_are_saved(short_fit, tmp_path: Path) -> None:
    fit, _ = short_fit
    chains = plot_background_chains(fit, output_path=tmp_path / "chains.png")
    corner_fig = plot_background_corner(
        fit, ["N_2nu", "c0"], output_path=tmp_path / "corner.png"
    )

    assert len(chains.axes) == 2 * len(BACKGROUND_PARAMETERS)
    assert len(corner_fig.axes) == 4
    assert (tmp_path / "chains.png").stat().st_size > 0
    assert (tmp_path / "corner.png").stat().st_size > 0
    plt.close("all")


def test_default_parameters_skip_fixed_ones() -> None:
    """Fixed coefficients are written as constants and left out by default."""
    import xarray as xr  # noqa: PLC0415

    rng = np.random.default_rng(0)
    draws = {
        name: (("chain", "draw"), rng.lognormal(size=(2, 30)))
        for name in BACKGROUND_PARAMETERS
    }
    draws["c0"] = (("chain", "draw"), np.zeros((2, 30)))
    draws["cQ"] = (("chain", "draw"), np.zeros((2, 30)))
    posterior = {"posterior": xr.Dataset(draws)}

    assert posterior_draws(posterior).shape == (2, 30, len(BACKGROUND_PARAMETERS) - 2)
    chains = plot_background_chains(posterior)
    assert len(chains.axes) == 2 * (len(BACKGROUND_PARAMETERS) - 2)
    plt.close("all")

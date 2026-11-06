# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from bosonic_dm.fitting.background_fit import load_background_fit, load_stan_data
from bosonic_dm.fitting.model_comparison import check_bridge_sampling, pointwise_loo
from bosonic_dm.plotting.model_comparison import (
    _signed_loo,
    plot_bf_prior_sensitivity,
    plot_bridge_diagnostics,
    plot_linear_prior_posterior,
    plot_model_difference,
    plot_pointwise_elpd,
    plot_ppc_discrepancy,
    plot_shared_parameters,
)

# Short toy fits can have a few bins with high Pareto k; that warning is real
# diagnostic output, not a failure of the plots.
pytestmark = pytest.mark.filterwarnings("ignore:Estimated shape parameter of Pareto")


def _record(group_bf: float, elpd_diff: float, best: str) -> dict:
    """A ``model_comparison.yaml`` record shaped like the notebook output."""
    worse = "full" if best == "no linear" else "no linear"
    return {
        "bayes_factor": {"model_a": "full", "model_b": "no linear", "log_bf": group_bf},
        "loo": [
            {"model": best, "rank": 0, "elpd_diff": 0.0, "dse": 0.0, "p_worse": None},
            {"model": worse, "rank": 1, "elpd_diff": elpd_diff, "dse": 0.28},
        ],
        "prior_sensitivity": [
            {"c0_cQ_prior_sd": sd, "log_bf": group_bf + shift, "log_bf_sd": 0.02}
            for sd, shift in ((2.0, -0.14), (0.5, 0.06), (1.0, 0.0))
        ],
    }


def test_signed_loo_difference() -> None:
    assert _signed_loo(_record(-0.38, -0.4, best="no linear")) == (-0.4, 0.28)
    assert _signed_loo(_record(2.0, -1.5, best="full")) == (1.5, 0.28)


def test_plot_bf_prior_sensitivity(tmp_path: Path) -> None:
    records = {
        "ICPC group1": _record(-0.38, -0.4, best="no linear"),
        "BEGe": _record(4.0, -3.1, best="full"),
    }
    output = tmp_path / "bf.png"

    fig = plot_bf_prior_sensitivity(records, output_path=output)

    assert len(fig.axes) == 2
    assert output.exists()
    plt.close(fig)


def test_plot_model_difference(saved_toy_fits: dict, tmp_path: Path) -> None:
    full = load_background_fit(saved_toy_fits["full"])
    no_linear = load_background_fit(saved_toy_fits["no_linear"])
    output = tmp_path / "difference.png"

    fig = plot_model_difference(
        full, no_linear, full.counts, window_keV=60.0, output_path=output
    )

    assert len(fig.axes) == 3
    assert output.exists()
    plt.close(fig)


def test_plot_pointwise_elpd(toy_fits: dict) -> None:
    (fit_a, data), (fit_b, _) = toy_fits["full"], toy_fits["no_linear"]

    fig = plot_pointwise_elpd(fit_a, fit_b, data)

    cumulative = fig.axes[1].lines[-1].get_ydata()
    total = pointwise_loo(fit_a, data)[0].sum() - pointwise_loo(fit_b, data)[0].sum()
    assert cumulative[-1] == pytest.approx(total)
    assert len(fig.axes) == 3
    plt.close(fig)


def test_plot_linear_prior_posterior(saved_toy_fits: dict) -> None:
    tree = xr.load_datatree(saved_toy_fits["full"] / "posterior.nc")
    data = load_stan_data(saved_toy_fits["full"])

    fig = plot_linear_prior_posterior(tree, data, sensitivity_posteriors={2.0: tree})

    assert len(fig.axes) == 4
    plt.close(fig)
    with pytest.raises(ValueError, match="c0 and cQ"):
        plot_linear_prior_posterior(tree, load_stan_data(saved_toy_fits["no_linear"]))


def test_plot_shared_parameters(toy_fits: dict) -> None:
    fig = plot_shared_parameters(toy_fits["full"][0], toy_fits["no_linear"][0])

    assert len(fig.axes) == 7  # nine parameters minus c0 and cQ
    plt.close(fig)


def test_plot_bridge_diagnostics(saved_toy_fits: dict) -> None:
    tree = xr.load_datatree(saved_toy_fits["no_linear"] / "posterior.nc")
    check = check_bridge_sampling(
        tree, load_stan_data(saved_toy_fits["no_linear"]), seeds=(1, 2)
    )

    fig = plot_bridge_diagnostics(check)

    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 3 + len(check.parameters)
    plt.close(fig)


def test_plot_ppc_discrepancy(toy_fits: dict, saved_toy_fits: dict) -> None:
    fit, data = toy_fits["full"]
    tree = xr.load_datatree(saved_toy_fits["no_linear"] / "posterior.nc")

    fig = plot_ppc_discrepancy(fit, tree, data, max_draws=300)

    assert len(fig.axes) == 3
    z_points = [
        line.get_ydata() for line in fig.axes[2].lines if line.get_marker() == "o"
    ]
    assert len(z_points) == 2
    assert all(np.isfinite(z).any() for z in z_points)
    plt.close(fig)

# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from bosonic_dm.fitting.sensitivity import (
    BAND_QUANTILES,
    draw_background_toys,
    expected_limits,
    stack_backgrounds,
    window_slice,
)
from bosonic_dm.fitting.signal_fit import rate_posterior_quantiles, rate_upper_limits
from bosonic_dm.fitting.signal_template import (
    BackgroundPrediction,
    GroupEfficiency,
    build_signal_window,
    gaussian_bin_probabilities,
)
from bosonic_dm.plotting.sensitivity import plot_expected_limits

# --- grid posterior ---------------------------------------------------------

EDGES = np.arange(488.0, 513.0)
B_LEVELS = np.array([5.0, 2.0])


def _arrays(
    sigmas: tuple[float, float] = (1.0, 1.4), exposures: tuple[float, float] = (10, 4)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    template = np.vstack(
        [
            exposure * 0.9 * gaussian_bin_probabilities(EDGES, 500.0, sigma)
            for sigma, exposure in zip(sigmas, exposures, strict=True)
        ]
    )
    b_hat = np.repeat(B_LEVELS[:, None], len(EDGES) - 1, axis=1)
    use_bin = np.ones_like(b_hat, dtype=bool)
    use_bin[0, 5:8] = False
    return b_hat, template, use_bin


def _brute_force(
    counts: np.ndarray,
    b_hat: np.ndarray,
    template: np.ndarray,
    use_bin: np.ndarray,
    quantiles: tuple[float, ...],
    rate_max: float,
    n_points: int = 400_001,
) -> np.ndarray:
    rates = np.linspace(0.0, rate_max, n_points)
    expected = b_hat[use_bin] + rates[:, None] * template[use_bin]
    log_like = (counts[use_bin] * np.log(expected) - expected).sum(axis=1)
    cdf = cumulative_trapezoid(np.exp(log_like - log_like.max()), rates, initial=0)
    return np.interp(quantiles, cdf / cdf[-1], rates)


def test_zero_background_limit_is_analytic() -> None:
    # Posterior exp(-R S): R_q S = -ln(1 - q).
    shape = (1, 1)
    args = (np.zeros(shape), np.full(shape, 1e-12), np.full(shape, 2.0))
    quantiles = rate_posterior_quantiles(*args, np.ones(shape, bool), (0.5, 0.9))[0]
    np.testing.assert_allclose(quantiles * 2.0, [np.log(2), -np.log(0.1)], rtol=1e-4)


def test_grid_stops_when_the_posterior_does_not_fall() -> None:
    shape = (1, 1)
    args = (np.zeros(shape), np.full(shape, 1e-12), np.full(shape, 2.0))
    with pytest.raises(RuntimeError, match="did not fall"):
        rate_upper_limits(*args, np.ones(shape, bool), max_doublings=0)


def test_high_background_limit_matches_gaussian_approximation() -> None:
    # With R_hat ~ N(0, sigma) and R >= 0, the median toy has R_hat = 0 and
    # R90 = 1.645 sigma (half-normal posterior); sigma = sqrt(B) / S.
    b_hat, template, use_bin = np.full((1, 1), 1e4), np.ones((1, 1)), np.ones((1, 1))
    sigma = 100.0
    toys = draw_background_toys(b_hat, 2000, base_seed=4)
    limits = rate_upper_limits(toys, b_hat, template, use_bin)
    asimov = rate_upper_limits(b_hat, b_hat, template, use_bin)[0]
    assert np.median(limits) == pytest.approx(1.645 * sigma, rel=0.03)
    assert asimov == pytest.approx(1.645 * sigma, rel=0.02)


@pytest.mark.parametrize(
    ("seed", "injected_rate", "extra_peak_counts"),
    [(3, 0.0, 0), (4, 0.0, 0), (5, 30.0, 0), (6, 30.0, 200)],
)
def test_grid_matches_brute_force(
    seed: int, injected_rate: float, extra_peak_counts: int
) -> None:
    b_hat, template, use_bin = _arrays()
    counts = np.random.default_rng(seed).poisson(b_hat + injected_rate * template)
    counts[0, 12] += extra_peak_counts
    quantiles = (0.05, 0.5, 0.9)
    grid = rate_posterior_quantiles(counts, b_hat, template, use_bin, quantiles)[0]
    expected = _brute_force(counts, b_hat, template, use_bin, quantiles, 3 * grid[-1])
    np.testing.assert_allclose(grid, expected, rtol=2e-3)


def test_masked_bins_do_not_change_the_limit() -> None:
    b_hat, template, use_bin = _arrays()
    counts = np.random.default_rng(1).poisson(b_hat)
    peaked = counts.copy()
    peaked[~use_bin] += 1000
    b_masked = b_hat.copy()
    b_masked[~use_bin] = 0.0
    reference = rate_upper_limits(counts, b_hat, template, use_bin)
    np.testing.assert_allclose(
        rate_upper_limits(peaked, b_masked, template, use_bin), reference, rtol=1e-12
    )


def test_datasets_are_independent_of_their_neighbours() -> None:
    b_hat, template, use_bin = _arrays()
    toys = draw_background_toys(b_hat, 20, base_seed=2)
    all_at_once = rate_upper_limits(toys, b_hat, template, use_bin, memory_bytes=1)
    one_by_one = [rate_upper_limits(t, b_hat, template, use_bin)[0] for t in toys]
    np.testing.assert_allclose(all_at_once, one_by_one, rtol=1e-12)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"b_hat": np.zeros((2, 24))}, "b_hat must be finite and positive"),
        ({"use_bin": np.zeros((2, 24), bool)}, "At least one bin"),
        ({"signal_template": np.zeros((2, 24))}, "template is zero"),
        ({"counts": -np.ones((2, 24))}, "counts must be finite"),
        ({"counts": np.ones((3, 24))}, "counts must have shape"),
        ({"quantiles": (1.0,)}, "quantiles"),
    ],
)
def test_rate_posterior_rejects_invalid_inputs(change: dict, match: str) -> None:
    b_hat, template, use_bin = _arrays()
    inputs = {
        "counts": np.ones((2, 24)),
        "b_hat": b_hat,
        "signal_template": template,
        "use_bin": use_bin,
        "quantiles": (0.9,),
    }
    inputs.update(change)
    with pytest.raises(ValueError, match=match):
        rate_posterior_quantiles(**inputs)


def test_injected_signal_is_recovered_with_nominal_coverage() -> None:
    true_rate = 20.0
    b_hat, template, use_bin = _arrays()
    rng = np.random.default_rng(8)
    counts = rng.poisson(b_hat + true_rate * template, size=(400, *b_hat.shape))
    quantiles = rate_posterior_quantiles(counts, b_hat, template, use_bin, (0.5, 0.9))
    medians, limits = quantiles.T
    assert abs(medians.mean() - true_rate) < 3 * medians.std(ddof=1) / np.sqrt(400)
    assert 0.85 < np.mean(limits > true_rate) < 0.95


# --- pseudo-experiments -----------------------------------------------------


def test_toys_are_reproducible_and_independent_of_the_toy_count() -> None:
    b_hat = np.array([[3.0, 0.0, 10.0], [1.0, 2.0, 3.0]])
    toys = draw_background_toys(b_hat, 10, base_seed=7)
    assert toys.shape == (10, 2, 3)
    assert toys.dtype == np.int64
    np.testing.assert_array_equal(toys, draw_background_toys(b_hat, 10, base_seed=7))
    np.testing.assert_array_equal(
        toys[4:6], draw_background_toys(b_hat, 2, base_seed=7, first_toy=4)
    )
    assert not np.array_equal(toys, draw_background_toys(b_hat, 10, base_seed=8))
    assert np.all(toys[:, 0, 1] == 0)


def test_toys_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="b_hat"):
        draw_background_toys(np.array([-1.0]), 1, base_seed=0)
    with pytest.raises(ValueError, match="n_toys"):
        draw_background_toys(np.array([1.0]), 0, base_seed=0)


def test_window_slice() -> None:
    edges = np.arange(400.0, 601.0)
    assert window_slice(edges, np.arange(488.0, 513.0)) == slice(88, 112)
    with pytest.raises(ValueError, match="contiguous"):
        window_slice(edges, np.array([488.0, 489.5]))


def _det(a: float, b: float, expo: float, usability: str = "on") -> dict:
    return {
        "usability": usability,
        "expo": expo,
        "a": a,
        "b": b,
        "a_unc": 0.0,
        "b_unc": 0.0,
        "ab_corr": 0.0,
    }


ERES = {
    "p03": {"r000": {"D1": _det(1.0, 0.002, 2.0), "D2": _det(2.0, 0.003, 1.0)}},
    "p04": {"r000": {"D2": _det(2.0, 0.003, 1.0), "D4": _det(1.5, 0.002, 6.0)}},
}
GROUPS = {"A": {"D1": "all", "D2": "all"}, "B": {"D4": "all"}}
EFFICIENCIES = {
    "A": GroupEfficiency(
        np.array([200.0, 1000.0]),
        np.array([0.8, 0.9]),
        np.array([0.01, 0.01]),
        np.array([4.0, 4.0]),
    ),
    "B": GroupEfficiency(
        np.array([200.0, 1000.0]),
        np.array([0.5, 0.5]),
        np.array([0.01, 0.01]),
        np.array([6.0, 6.0]),
    ),
}


def _background(group: str, level: float, masked: tuple = ()) -> BackgroundPrediction:
    edges = np.arange(400.0, 601.0)
    used = np.ones(len(edges) - 1, dtype=bool)
    for low, high in masked:
        used &= (edges[1:] <= low) | (edges[:-1] >= high)
    return BackgroundPrediction(
        group=group,
        bin_edges=edges,
        counts=np.full(len(edges) - 1, round(level), dtype=np.int64),
        b_hat=np.full(len(edges) - 1, level),
        used_in_fit=used,
        passed_diagnostics=group == "A",
        files_sha256={"bhat.parquet": group},
    )


BACKGROUNDS = {
    "A": _background("A", 6.5, masked=((500.0, 530.0),)),
    "B": _background("B", 3.0, masked=((500.0, 530.0),)),
}
WINDOW_OPTIONS = {
    "eres_dict": ERES,
    "groups_dict": GROUPS,
    "group_efficiencies": EFFICIENCIES,
}
MASSES = (450.0, 495.0, 515.0)


@pytest.fixture(scope="module")
def limits() -> object:
    return expected_limits(
        MASSES, BACKGROUNDS, n_toys=40, base_seed=11, n_bootstrap=50, **WINDOW_OPTIONS
    )


def test_expected_limits_tables(limits: object) -> None:
    summary = limits.summary
    assert summary["mass_keV"].to_list() == list(MASSES)
    assert limits.per_toy.shape == (len(MASSES) * 40, 3)
    assert summary["near_masked_line"].to_list() == [False, True, True]
    assert summary["all_bins_masked"].to_list() == [False, False, True]

    finite = summary.filter(~summary["all_bins_masked"])
    bands = finite.select(list(BAND_QUANTILES)).to_numpy()
    assert np.all(np.diff(bands, axis=1) >= 0)
    assert np.all(finite["r90_median_mcse"].to_numpy() > 0)
    assert finite["signal_fraction_in_fit_bins"][0] == pytest.approx(1.0)
    assert 0 < finite["signal_fraction_in_fit_bins"][1] < 1

    masked = summary.filter(summary["all_bins_masked"])
    assert np.isnan(masked["r90_median"][0])
    masked_toys = limits.per_toy.filter(limits.per_toy["mass_keV"] == 515.0)
    assert masked_toys["r90"].is_nan().all()

    assert limits.settings["n_toys"] == 40
    assert limits.settings["groups"]["B"]["passed_diagnostics"] is False
    assert limits.settings["toys_given"] is False


def test_expected_limits_reuse_each_toy_across_masses(limits: object) -> None:
    # The 450 keV limits equal a direct grid fit of the same toys.
    edges, b_hat_full = stack_backgrounds(BACKGROUNDS)
    toys = draw_background_toys(b_hat_full, 40, base_seed=11)
    window = build_signal_window(450.0, BACKGROUNDS, **WINDOW_OPTIONS)
    bins = window_slice(edges, window.bin_edges)
    direct = rate_upper_limits(
        toys[:, :, bins],
        window.stacked("b_hat"),
        window.stacked("signal_template"),
        window.stacked("use_bin"),
    )
    per_toy = limits.per_toy.filter(limits.per_toy["mass_keV"] == 450.0)
    np.testing.assert_allclose(per_toy["r90"].to_numpy(), direct, rtol=1e-12)


def test_expected_limits_are_reproducible(limits: object) -> None:
    fewer = expected_limits(
        MASSES, BACKGROUNDS, n_toys=10, base_seed=11, n_bootstrap=50, **WINDOW_OPTIONS
    )
    first = limits.per_toy.filter(limits.per_toy["toy"] < 10)
    np.testing.assert_allclose(
        fewer.per_toy["r90"].to_numpy(), first["r90"].to_numpy(), rtol=0
    )


def test_expected_limits_accept_given_toys() -> None:
    _, b_hat_full = stack_backgrounds(BACKGROUNDS)
    asimov = np.repeat(b_hat_full[None], 3, axis=0)
    result = expected_limits(
        (450.0,), BACKGROUNDS, toys=asimov, n_bootstrap=0, **WINDOW_OPTIONS
    )
    row = result.summary.row(0, named=True)
    assert result.settings["toys_given"] is True
    assert row["r90_median"] == pytest.approx(row["r90_asimov"], rel=1e-12)
    with pytest.raises(ValueError, match="toys must have shape"):
        expected_limits((450.0,), BACKGROUNDS, toys=asimov[:, :1], **WINDOW_OPTIONS)


# --- plots ------------------------------------------------------------------


def test_plot_expected_limits(limits: object, tmp_path: Path) -> None:
    path = tmp_path / "brazil.png"
    fig = plot_expected_limits(limits.summary, title="toy", output_path=path)
    assert path.exists()
    assert fig.axes[0].get_yscale() == "log"
    plt.close(fig)

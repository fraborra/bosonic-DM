# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from bosonic_dm.efficiency import efficiency_table
from bosonic_dm.fitting.signal_template import (
    FWHM_TO_SIGMA,
    BackgroundPrediction,
    GroupEfficiency,
    build_signal_window,
    calibration_offset,
    gaussian_bin_probabilities,
    group_centroid_offset,
    group_efficiency,
    group_fwhm,
    interpolate_efficiency,
    load_background_prediction,
)
from bosonic_dm.stats import bayesian_efficiency
from bosonic_dm.yaml_io import write_yaml


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


# D1 belongs to group A only in p03; D3 is switched off; D4 is in group B.
ERES = {
    "p03": {
        "r000": {
            "D1": _det(1.0, 0.002, 2.0),
            "D2": _det(2.0, 0.003, 1.0),
            "D3": _det(9.0, 0.009, 5.0, usability="off"),
            "D4": _det(1.5, 0.002, 3.0),
        }
    },
    "p04": {
        "r000": {
            "D1": _det(4.0, 0.004, 4.0),
            "D2": _det(2.0, 0.003, 1.0),
            "D4": _det(1.5, 0.002, 3.0),
        }
    },
}
GROUPS = {
    "A": {"D1": {"p03": "all"}, "D2": "all", "D3": "all"},
    "B": {"D4": "all"},
}


def _efficiency_entry(efficiency: float, expo: float) -> dict:
    # 1000 primaries per unit of exposure, so primaries follow exposure.
    n_primaries = round(1000 * expo)
    return {
        "expo": expo,
        "n_primaries": n_primaries,
        "selections": {
            "all": {
                "status": "valid",
                "n_events": round(efficiency * n_primaries),
                "efficiency": efficiency,
                "efficiency_stat_unc": 0.01,
                "efficiency_syst_fwhm": 0.0,
            }
        },
    }


def _efficiency_dict() -> dict:
    output = {}
    for energy, scale in ((200, 0.8), (400, 0.9)):
        output[energy] = {
            "D1": {
                "period_runs": {
                    "p03": {"r000": _efficiency_entry(scale, 2.0)},
                    "p04": {"r000": _efficiency_entry(0.1, 4.0)},  # not in group A
                }
            },
            "D2": {
                "period_runs": {
                    "p03": {"r000": _efficiency_entry(scale - 0.2, 1.0)},
                    "p04": {"r000": _efficiency_entry(scale - 0.2, 1.0)},
                }
            },
            "D4": {
                "period_runs": {
                    "p03": {"r000": _efficiency_entry(0.5, 3.0)},
                    "p04": {"r000": _efficiency_entry(0.5, 3.0)},
                }
            },
        }
    return output


def test_gaussian_bin_probabilities_match_normal_cdf() -> None:
    edges = np.array([480.0, 490.0, 497.5, 499.0, 500.0, 500.3, 503.0, 520.0])
    expected = np.diff(norm.cdf(edges, loc=500.0, scale=1.3))
    np.testing.assert_allclose(
        gaussian_bin_probabilities(edges, 500.0, 1.3), expected, rtol=1e-12, atol=1e-16
    )


def test_gaussian_upper_tail_has_no_cancellation() -> None:
    edges = np.array([10.0, 11.0, 12.0])
    expected = -np.diff(norm.sf(edges))
    np.testing.assert_allclose(
        gaussian_bin_probabilities(edges, 0.0, 1.0), expected, rtol=1e-10
    )
    assert np.all(gaussian_bin_probabilities(edges, 0.0, 1.0) > 0)


def test_gaussian_bin_probabilities_sum_to_one() -> None:
    edges = np.linspace(400.0, 600.0, 201)
    assert gaussian_bin_probabilities(edges, 500.0, 1.0).sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("edges", "sigma"), [([1.0, 1.0, 2.0], 1.0), ([1.0], 1.0), ([1.0, 2.0], 0.0)]
)
def test_gaussian_bin_probabilities_reject_invalid_input(
    edges: list[float], sigma: float
) -> None:
    with pytest.raises(ValueError):
        gaussian_bin_probabilities(edges, 0.0, sigma)


def test_group_fwhm_uses_period_membership_and_exposure_weights() -> None:
    energy = 500.0
    # Group A: D1 in p03 (weight 2), D2 in p03 and p04 (weight 1 each).
    fwhm = np.sqrt(
        np.array([1.0 + 0.002 * energy, 2.0 + 0.003 * energy, 2.0 + 0.003 * energy])
    )
    weights = np.array([2.0, 1.0, 1.0])
    mean, unc = group_fwhm(ERES, GROUPS["A"], energy)
    assert mean == pytest.approx(np.average(fwhm, weights=weights))
    assert unc > 0  # weighted scatter between detectors

    mean_b, unc_b = group_fwhm(ERES, GROUPS["B"], energy)
    assert mean_b == pytest.approx(np.sqrt(1.5 + 0.002 * energy))
    assert unc_b == pytest.approx(0.0)


def test_exposure_key_selects_the_weights_and_the_group_exposure() -> None:
    alternative = {"D1": 1.0, "D2": 2.0, "D3": 5.0, "D4": 3.0}
    eres = {
        period: {
            run: {
                detector: {**info, "expo_alt": alternative[detector]}
                for detector, info in detectors.items()
            }
            for run, detectors in runs.items()
        }
        for period, runs in ERES.items()
    }
    energy = 500.0
    fwhm = np.sqrt(
        np.array([1.0 + 0.002 * energy, 2.0 + 0.003 * energy, 2.0 + 0.003 * energy])
    )

    mean, _ = group_fwhm(eres, GROUPS["A"], energy, exposure_key="expo_alt")
    assert mean == pytest.approx(np.average(fwhm, weights=[1.0, 2.0, 2.0]))

    window = build_signal_window(
        energy,
        {"A": _background("A")},
        eres_dict=eres,
        groups_dict=GROUPS,
        group_efficiencies=_efficiencies(),
        exposure_key="expo_alt",
    )
    assert window.groups["A"].exposure_kg_yr == pytest.approx(5.0)
    assert window.groups["A"].fwhm_keV == pytest.approx(mean)


def test_group_fwhm_without_entries_raises() -> None:
    with pytest.raises(ValueError):
        group_fwhm(ERES, {"D3": "all"}, 500.0)


def test_group_efficiency_respects_period_membership() -> None:
    efficiencies = group_efficiency(_efficiency_dict(), GROUPS)

    group_a = efficiencies["A"]
    np.testing.assert_array_equal(group_a.energies_keV, [200.0, 400.0])
    # D1 p03 (0.8, weight 2) and D2 p03/p04 (0.6, weight 1 each); D1 p04 excluded.
    # Counts: 1600/2000 + 600/1000 + 600/1000, with one prior for the group.
    assert group_a.efficiency[0] == pytest.approx(bayesian_efficiency(2800, 4000)[0])
    assert group_a.efficiency[0] == pytest.approx((2 * 0.8 + 2 * 0.6) / 4, abs=1e-4)
    np.testing.assert_allclose(group_a.exposure_kg_yr, [4.0, 4.0])
    np.testing.assert_allclose(efficiencies["B"].efficiency, [0.5, 0.5])


def test_group_efficiency_accepts_the_efficiency_table() -> None:
    from_nested = group_efficiency(_efficiency_dict(), GROUPS)
    from_table = group_efficiency(efficiency_table(_efficiency_dict()), GROUPS)

    assert list(from_table) == list(from_nested)
    for group, expected in from_nested.items():
        for field in ("energies_keV", "efficiency", "efficiency_unc", "exposure_kg_yr"):
            np.testing.assert_allclose(
                getattr(from_table[group], field), getattr(expected, field)
            )


def test_group_efficiency_uncertainty_comes_from_the_combined_counts() -> None:
    group_a = group_efficiency(_efficiency_dict(), GROUPS)["A"]
    # 2800 events in 4000 primaries; the run-level uncertainties are not used.
    assert group_a.efficiency_unc[0] == pytest.approx(
        bayesian_efficiency(2800, 4000)[1]
    )


def test_group_efficiency_rejects_detector_only_results() -> None:
    old_format = {200: {"D1": {"expo": 1.0, "selections": {"all": {}}}}}
    with pytest.raises(ValueError, match="period_runs"):
        group_efficiency(old_format, GROUPS)


def test_interpolate_efficiency() -> None:
    assert interpolate_efficiency([200, 400], [0.8, 0.9], 300.0) == pytest.approx(0.85)
    assert interpolate_efficiency([200, 400], [0.8, 0.9], 400.0) == pytest.approx(0.9)
    with pytest.raises(ValueError, match="extrapolation"):
        interpolate_efficiency([200, 400], [0.8, 0.9], 401.0)


def _background(
    group: str, start: float = 400.0, stop: float = 600.0, masked: tuple = ()
) -> BackgroundPrediction:
    edges = np.arange(start, stop + 1.0, 1.0)
    used = np.ones(len(edges) - 1, dtype=bool)
    for low, high in masked:
        used &= (edges[1:] <= low) | (edges[:-1] >= high)
    return BackgroundPrediction(
        group=group,
        bin_edges=edges,
        counts=np.full(len(edges) - 1, 7, dtype=np.int64),
        b_hat=np.full(len(edges) - 1, 6.5),
        used_in_fit=used,
        passed_diagnostics=True,
        files_sha256={"bhat.parquet": group},
    )


def _efficiencies() -> dict[str, GroupEfficiency]:
    return {
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


def _window(mass: float = 500.0, **backgrounds: BackgroundPrediction):
    backgrounds = backgrounds or {"A": _background("A"), "B": _background("B")}
    return build_signal_window(
        mass,
        backgrounds,
        eres_dict=ERES,
        groups_dict=GROUPS,
        group_efficiencies=_efficiencies(),
    )


def test_signal_window_template_normalization() -> None:
    window = _window(500.0)

    np.testing.assert_array_equal(window.bin_edges, np.arange(488.0, 513.0))
    assert window.group_names == ["A", "B"]
    assert window.stacked("signal_template").shape == (2, 24)

    group_a = window.groups["A"]
    fwhm, _ = group_fwhm(ERES, GROUPS["A"], 500.0)
    efficiency = 0.8 + 0.1 * 300.0 / 800.0
    containment = norm.cdf(12.0 / (fwhm * FWHM_TO_SIGMA)) * 2 - 1
    assert group_a.exposure_kg_yr == pytest.approx(4.0)
    assert group_a.efficiency == pytest.approx(efficiency)
    assert group_a.sigma_keV == pytest.approx(fwhm * FWHM_TO_SIGMA)
    assert group_a.window_containment == pytest.approx(containment)
    assert group_a.signal_template.sum() == pytest.approx(
        4.0 * efficiency * containment
    )
    np.testing.assert_allclose(group_a.b_hat, 6.5)
    assert window.background_sha256 == {
        "A": {"bhat.parquet": "A"},
        "B": {"bhat.parquet": "B"},
    }


def test_signal_window_off_grid_mass_and_mask() -> None:
    window = _window(
        500.5,
        A=_background("A", masked=((495.0, 498.0),)),
        B=_background("B"),
    )
    np.testing.assert_array_equal(window.bin_edges, np.arange(488.0, 514.0))
    expected = ~((window.bin_edges[:-1] >= 495.0) & (window.bin_edges[1:] <= 498.0))
    np.testing.assert_array_equal(window.groups["A"].use_bin, expected)
    assert window.groups["B"].use_bin.all()


def test_signal_window_warns_on_exposure_mismatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    efficiencies = _efficiencies()
    efficiencies["B"] = GroupEfficiency(
        np.array([200.0, 1000.0]),
        np.array([0.5, 0.5]),
        np.array([0.01, 0.01]),
        np.array([3.0, 3.0]),
    )
    build_signal_window(
        500.0,
        {"A": _background("A"), "B": _background("B")},
        eres_dict=ERES,
        groups_dict=GROUPS,
        group_efficiencies=efficiencies,
    )
    assert "differs from the data exposure" in caplog.text


@pytest.mark.parametrize(
    ("backgrounds", "match"),
    [
        ({"A": _background("A"), "B": _background("B", stop=650.0)}, "Bin edges"),
        ({"A": _background("A", stop=505.0)}, "outside"),
        ({"A": _background("A", masked=((480.0, 520.0),))}, "masked"),
    ],
)
def test_signal_window_rejects_invalid_inputs(backgrounds: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _window(500.0, **backgrounds)


def test_load_background_prediction(tmp_path: Path) -> None:
    edges = np.arange(60.0, 66.0)
    pl.DataFrame(
        {
            "bin_lo_keV": edges[:-1],
            "bin_hi_keV": edges[1:],
            "counts": [3, 4, 5, 6, 7],
            "used_in_fit": [True, False, True, True, True],
            "B_hat": [3.1, 4.2, 5.0, 6.1, 7.2],
        }
    ).write_parquet(tmp_path / "bhat.parquet")
    write_yaml(
        {"passed_diagnostics": True, "provenance": {"group": "ICPC group1"}},
        tmp_path / "summary.yaml",
    )

    background = load_background_prediction(tmp_path)
    assert background.group == "ICPC group1"
    np.testing.assert_array_equal(background.bin_edges, edges)
    np.testing.assert_array_equal(background.counts, [3, 4, 5, 6, 7])
    np.testing.assert_array_equal(background.used_in_fit, [1, 0, 1, 1, 1])
    assert background.passed_diagnostics
    assert set(background.files_sha256) == {"bhat.parquet", "summary.yaml"}


def _with_residuals(
    peaks: list[float], residuals: list[float], unc: list[float]
) -> dict:
    return {
        "cal_peaks_keV": peaks,
        "cal_residuals_keV": residuals,
        "cal_residuals_unc_keV": unc,
    }


# Residuals added to every usable entry except D2 in p04.
RESIDUALS = {
    period: {
        run: {
            det: (
                info
                if (period, det) == ("p04", "D2")
                else {
                    **info,
                    **_with_residuals(
                        [238.6, 583.2, 2614.5],
                        [0.1 * (i + 1), 0.3 * (i + 1), -0.2],
                        [0.02, 0.02, 0.04],
                    ),
                }
            )
            for i, (det, info) in enumerate(detectors.items())
        }
        for run, detectors in runs.items()
    }
    for period, runs in ERES.items()
}


def test_calibration_offset_interpolates_and_flags_extrapolation() -> None:
    # Unsorted input is sorted internally.
    peaks, residuals = [583.2, 238.6, 2614.5], [0.3, 0.1, -0.2]
    offset, unc, outside = calibration_offset(peaks, residuals, 410.9, [0.02] * 3)
    assert offset == pytest.approx(0.2)
    assert unc == pytest.approx(0.02)
    assert not outside

    offset, _, outside = calibration_offset(peaks, residuals, 200.0)
    assert offset == pytest.approx(0.1)
    assert outside


@pytest.mark.parametrize(
    ("peaks", "residuals"),
    [([1.0, 2.0], [0.1]), ([np.nan], [0.1]), ([1.0, 1.0], [0.1, 0.2])],
)
def test_calibration_offset_rejects_invalid_curves(
    peaks: list[float], residuals: list[float]
) -> None:
    with pytest.raises(ValueError):
        calibration_offset(peaks, residuals, 500.0)


def test_group_centroid_offset_weights_and_missing_entries(
    caplog: pytest.LogCaptureFixture,
) -> None:
    energy = 410.9  # halfway between 238.6 and 583.2
    offset, unc, outside = group_centroid_offset(RESIDUALS, GROUPS["A"], energy)
    # Group A contributions: D1 p03 (index 0, weight 2) and D2 p03 (index 1,
    # weight 1); D2 p04 has no residuals and D3 is switched off.
    d1, d2 = 0.2 * 1, 0.2 * 2
    assert offset == pytest.approx((2 * d1 + 1 * d2) / 3)
    assert unc > 0
    assert not outside
    assert "No calibration residuals for D2 in p04-r000" in caplog.text
    assert "25.00% of the group exposure" in caplog.text

    with pytest.raises(ValueError, match="calibration residuals"):
        group_centroid_offset(ERES, GROUPS["A"], energy)


def test_signal_window_applies_calibration_offset() -> None:
    nominal = _window(500.0)
    shifted = build_signal_window(
        500.0,
        {"A": _background("A"), "B": _background("B")},
        eres_dict=ERES,
        groups_dict=GROUPS,
        group_efficiencies=_efficiencies(),
        calibration_residuals=RESIDUALS,
    )
    np.testing.assert_array_equal(shifted.bin_edges, nominal.bin_edges)
    for name in ("A", "B"):
        group = shifted.groups[name]
        expected, expected_unc, _ = group_centroid_offset(
            RESIDUALS, GROUPS[name], 500.0
        )
        assert nominal.groups[name].centroid_offset_keV == 0.0
        assert group.centroid_offset_keV == pytest.approx(expected)
        assert group.centroid_offset_unc_keV == pytest.approx(expected_unc)
        assert group.centroid_keV == pytest.approx(500.0 + expected)
        np.testing.assert_allclose(
            group.signal_template,
            group.exposure_kg_yr
            * group.efficiency
            * gaussian_bin_probabilities(
                shifted.bin_edges, 500.0 + expected, group.sigma_keV
            ),
        )

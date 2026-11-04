# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from bosonic_dm.background import prepare_background_spectra, spectrum_in_range


def _frame(energies: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "energy": energies,
            "passes_analysis": [True] * len(energies),
            "passes_comparison": [True] * len(energies),
            "detector_name": ["A"] * len(energies),
            "period": ["p03"] * len(energies),
            "run": ["r000"] * len(energies),
        }
    )


def _exposures(value: float = 2.0) -> dict:
    return {"p03": {"r000": {"A": {"usability": "on", "expo": value}}}}


@pytest.mark.parametrize("lazy", [False, True])
def test_group_spectra_preserve_run_membership_and_zero_event_exposure(
    lazy: bool,
) -> None:
    frame = _frame([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0]).with_columns(
        pl.Series("detector_name", ["A", "A", "A", "A", "B", "B", "C", "A"]),
        pl.Series("period", ["p03", "p03", "p04", "p04", "p03", "p04", "p03", "p03"]),
        pl.Series(
            "passes_analysis", [True, False, True, True, True, False, True, True]
        ),
        pl.Series(
            "passes_comparison", [True, True, False, True, False, True, True, True]
        ),
    )
    exposures = {
        "p03": {
            "r000": {
                "A": {"usability": "on", "expo": 2.0},
                "B": {"usability": "on", "expo": 3.0},
                "D": {"usability": "on", "expo": 4.0},
            }
        },
        "p04": {
            "r000": {
                "A": {"usability": "on", "expo": 5.0},
                "B": {"usability": "on", "expo": 7.0},
            }
        },
    }
    result = prepare_background_spectra(
        frame.lazy() if lazy else frame,
        {
            "first": {"A": {"p03": "all"}, "B": {"p04": "all"}},
            "second": {"A": {"p04": "all"}},
            "empty": {"D": "all"},
        },
        exposures,
        energy_range_keV=(0.0, 10.0),
        bin_width_keV=2.0,
    )

    first = result["first"]
    np.testing.assert_array_equal(first.energy, [1.0, 2.0, 6.0])
    np.testing.assert_array_equal(first["counts_analysis"], [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(first.counts_comparison, [1, 1, 0, 1, 0])
    assert first.expo == 9.0
    assert first.rate_analysis[0] == pytest.approx(1 / 18)
    np.testing.assert_array_equal(result["second"].counts_analysis, [0, 1, 1, 0, 0])
    assert result["second"].expo == 5.0
    assert result["empty"].expo == 4.0
    assert result["empty"].counts_analysis.sum() == 0
    assert first.fit_mask.all()  # Genuine empty bins remain available to fits.


def test_peak_exclusions_only_flag_bins_and_keep_counts_and_exposure() -> None:
    result = prepare_background_spectra(
        _frame([1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0]),
        {"group": {"A": "all"}},
        _exposures(),
        energy_range_keV=(0.0, 8.0),
        bin_width_keV=2.0,
        peak_centres_keV=[3.0],
        exclusion_width_keV=0.5,
    )["group"]

    np.testing.assert_array_equal(
        result.energy, [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0]
    )
    np.testing.assert_array_equal(result.counts_analysis, [1, 4, 2, 1])
    np.testing.assert_array_equal(result.fit_mask, [True, False, True, True])
    assert result.expo == 2.0


@pytest.mark.parametrize(
    ("edges", "expected_edges", "expected_rates"),
    [
        (None, [0.0, 2.0, 4.0, 5.0], [0.25, 0.25, 0.5]),
        ([0.0, 1.0, 4.0, 5.0], [0.0, 1.0, 4.0, 5.0], [0.5, 1 / 6, 0.5]),
    ],
)
def test_rates_use_actual_width_of_each_bin(
    edges: list[float] | None,
    expected_edges: list[float],
    expected_rates: list[float],
) -> None:
    spectrum = prepare_background_spectra(
        _frame([0.5, 2.5, 4.5, 5.0]),
        {"group": {"A": "all"}},
        _exposures(),
        energy_range_keV=(0.0, 5.0),
        bin_width_keV=2.0,
        bin_edges_keV=edges,
    )["group"]

    np.testing.assert_array_equal(spectrum.bin_edges, expected_edges)
    np.testing.assert_array_equal(spectrum.counts_analysis, [1, 1, 1])
    np.testing.assert_allclose(spectrum.rate_analysis, expected_rates)


def test_custom_columns_null_flags_and_exposure_field() -> None:
    frame = (
        _frame([1.0, 2.0])
        .rename(
            {
                "energy": "E",
                "passes_analysis": "standard",
                "passes_comparison": "relaxed",
            }
        )
        .with_columns(pl.Series("standard", [True, None], dtype=pl.Boolean))
    )
    exposures = {"p03": {"r000": {"A": {"usability": "on", "live_expo": 4.0}}}}
    spectrum = prepare_background_spectra(
        frame,
        {"group": {"A": "all"}},
        exposures,
        bin_edges_keV=[0.0, 3.0],
        energy_column="E",
        analysis_column="standard",
        comparison_column="relaxed",
        exposure_key="live_expo",
    )["group"]

    np.testing.assert_array_equal(spectrum.mask_analysis, [True, False])
    np.testing.assert_array_equal(spectrum.counts_analysis, [1])
    np.testing.assert_array_equal(spectrum.counts_comparison, [2])
    assert spectrum.expo == 4.0


def test_zero_exposure_is_rejected_before_rate_normalization() -> None:
    with pytest.raises(ValueError, match="finite positive exposure"):
        prepare_background_spectra(
            _frame([100.0]), {"group": {"A": "all"}}, _exposures(0.0)
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"energy_range_keV": (10.0, 0.0)},
        {"bin_width_keV": 0.0},
        {"bin_edges_keV": [0.0, 1.0, 1.0]},
        {"peak_centres_keV": [float("nan")]},
        {"exclusion_width_keV": -1.0},
    ],
)
def test_invalid_spectrum_configuration_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        prepare_background_spectra(
            _frame([100.0]), {"group": {"A": "all"}}, _exposures(), **kwargs
        )


def test_spectrum_in_range_slices_both_selections_and_preserves_exclusions() -> None:
    frame = _frame([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    frame = frame.with_columns(
        pl.Series("passes_analysis", [True, False] * 5),
        pl.Series("passes_comparison", [False, True] * 5),
    )
    source = prepare_background_spectra(
        frame,
        {"group": {"A": "all"}},
        _exposures(),
        energy_range_keV=(0.0, 10.0),
        bin_width_keV=2.0,
        peak_centres_keV=[3.0],
        exclusion_width_keV=0.5,
    )["group"]
    original_counts = source.counts_analysis.copy()
    original_mask = source.fit_mask.copy()

    result = spectrum_in_range(source, (2.0, 8.0))

    np.testing.assert_array_equal(result.energy, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    np.testing.assert_array_equal(
        result.mask_analysis, [True, False, True, False, True, False]
    )
    np.testing.assert_array_equal(result.counts_analysis, [1, 1, 1])
    np.testing.assert_array_equal(result.counts_comparison, [1, 1, 1])
    np.testing.assert_array_equal(result.bin_edges, [2.0, 4.0, 6.0, 8.0])
    np.testing.assert_array_equal(result.bin_centres, [3.0, 5.0, 7.0])
    np.testing.assert_array_equal(result.bin_widths_keV, [2.0, 2.0, 2.0])
    np.testing.assert_allclose(result.rate_analysis, [0.25, 0.25, 0.25])
    np.testing.assert_array_equal(result.fit_mask, [False, True, True])
    assert result.expo == source.expo
    np.testing.assert_array_equal(source.counts_analysis, original_counts)
    np.testing.assert_array_equal(source.fit_mask, original_mask)


def test_spectrum_in_range_keeps_complete_bins_and_genuine_empty_bins() -> None:
    source = prepare_background_spectra(
        _frame([1.5, 2.0, 3.0, 8.0, 8.5]),
        {"group": {"A": "all"}},
        _exposures(),
        energy_range_keV=(0.0, 10.0),
        bin_width_keV=2.0,
    )["group"]
    result = spectrum_in_range(source, (1.0, 9.0))

    np.testing.assert_array_equal(result.bin_edges, [2.0, 4.0, 6.0, 8.0])
    np.testing.assert_array_equal(result.energy, [2.0, 3.0])
    np.testing.assert_array_equal(result.counts_analysis, [2, 0, 0])
    np.testing.assert_allclose(result.rate_analysis, [0.5, 0.0, 0.0])
    assert result.fit_mask.all()


@pytest.mark.parametrize(
    "energy_range",
    [(-1.0, 8.0), (0.0, 11.0), (5.0, 5.0), (1.1, 1.2), (float("nan"), 8.0)],
)
def test_spectrum_in_range_rejects_invalid_or_unavailable_range(
    energy_range: tuple[float, float],
) -> None:
    source = prepare_background_spectra(
        _frame([1.0]),
        {"group": {"A": "all"}},
        _exposures(),
        energy_range_keV=(0.0, 10.0),
    )["group"]
    with pytest.raises(ValueError):
        spectrum_in_range(source, energy_range)


def test_spectrum_in_range_cannot_restore_events_after_narrowing() -> None:
    source = prepare_background_spectra(
        _frame([1.0, 5.0, 9.0]),
        {"group": {"A": "all"}},
        _exposures(),
        energy_range_keV=(0.0, 10.0),
    )["group"]
    narrowed = spectrum_in_range(source, (4.0, 8.0))

    with pytest.raises(ValueError, match="exceeds the prepared group range"):
        spectrum_in_range(narrowed, (0.0, 10.0))

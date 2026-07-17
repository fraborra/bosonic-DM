# Copyright (C) 2025 Francesco Borra
#

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from bosonic_dm.efficiency import (
    compute_efficiency_from_lazyframe,
    filter_valid_selection_efficiency,
)


def _channelmap() -> dict:
    return {
        "V00001A": SimpleNamespace(daq=SimpleNamespace(rawid=1)),
        "V00002A": SimpleNamespace(daq=SimpleNamespace(rawid=2)),
    }


def _resolution() -> dict:
    return {
        200: {
            "V00001A": {"fwhm": 1.0, "unc": 0.1, "expo": 1.0},
            "V00002A": {"fwhm": 1.0, "unc": 0.1, "expo": 1.0},
        }
    }


def test_efficiency_marks_missing_primaries_and_psd_unavailable() -> None:
    frame = pl.DataFrame(
        {
            "rawid": [1, 1, 2],
            "energy": [200.0, 201.0, 200.0],
            "sim_e": [200, 200, 200],
            "is_good_channel": [True, True, True],
            "has_aoe": [False, False, True],
            "is_single_site": [False, False, True],
        }
    ).lazy()

    result = compute_efficiency_from_lazyframe(
        lf=frame,
        eres_dict=_resolution(),
        simulated_energies=[200],
        chmap=_channelmap(),
        vertex_counts={200: {"V00001A": 10}},
    )

    first = result[200]["V00001A"]
    assert first["status"] == "valid"
    assert first["selections"]["all"]["status"] == "valid"
    assert first["selections"]["valid-psd"]["status"] == "psd-unavailable"
    assert first["selections"]["valid-psd"]["efficiency"] is None

    second = result[200]["V00002A"]
    assert second["status"] == "missing-primaries"
    assert second["selections"]["all"]["status"] == "missing-primaries"
    assert second["selections"]["all"]["efficiency"] is None


def test_efficiency_marks_counts_above_primaries_invalid() -> None:
    frame = pl.DataFrame(
        {
            "rawid": [1, 1],
            "energy": [200.0, 200.5],
            "sim_e": [200, 200],
            "is_good_channel": [True, True],
            "has_aoe": [True, True],
            "is_single_site": [True, True],
        }
    ).lazy()

    result = compute_efficiency_from_lazyframe(
        lf=frame,
        eres_dict=_resolution(),
        simulated_energies=[200],
        chmap=_channelmap(),
        vertex_counts={200: {"V00001A": 1}},
        selections=["all"],
    )

    selection = result[200]["V00001A"]["selections"]["all"]
    assert selection["status"] == "invalid-counts"
    assert selection["efficiency"] is None


def test_efficiency_honors_configured_selections() -> None:
    frame = pl.DataFrame(
        {
            "rawid": [1],
            "energy": [200.0],
            "sim_e": [200],
            "is_good_channel": [True],
            "has_aoe": [True],
            "is_single_site": [True],
        }
    ).lazy()

    result = compute_efficiency_from_lazyframe(
        lf=frame,
        eres_dict=_resolution(),
        simulated_energies=[200],
        chmap=_channelmap(),
        vertex_counts={200: {"V00001A": 10}},
        selections=["all"],
    )

    assert list(result[200]["V00001A"]["selections"]) == ["all"]


def test_zero_reconstructed_events_remain_a_valid_zero_efficiency() -> None:
    frame = pl.DataFrame(
        {
            "rawid": [1],
            "energy": [200.0],
            "sim_e": [200],
            "is_good_channel": [True],
            "has_aoe": [True],
            "is_single_site": [True],
        }
    ).lazy()
    result = compute_efficiency_from_lazyframe(
        lf=frame,
        eres_dict=_resolution(),
        simulated_energies=[200],
        chmap=_channelmap(),
        vertex_counts={
            200: {
                "V00001A": 10,
                "V00002A": 10,
            }
        },
    )

    second = result[200]["V00002A"]
    assert second["psd_available"] is None
    assert second["selections"]["all"]["status"] == "valid"
    assert second["selections"]["all"]["n_events"] == 0
    assert second["selections"]["all"]["efficiency_mle"] == 0.0
    assert second["selections"]["valid-psd"]["status"] == "valid"


def test_status_filter_excludes_unavailable_psd() -> None:
    frame = pl.DataFrame(
        {
            "rawid": [1, 2],
            "energy": [200.0, 200.0],
            "sim_e": [200, 200],
            "is_good_channel": [True, True],
            "has_aoe": [True, False],
            "is_single_site": [True, False],
        }
    ).lazy()
    result = compute_efficiency_from_lazyframe(
        lf=frame,
        eres_dict=_resolution(),
        simulated_energies=[200],
        chmap=_channelmap(),
        vertex_counts={
            200: {
                "V00001A": 10,
                "V00002A": 10,
            }
        },
    )

    filtered = filter_valid_selection_efficiency(result, "valid-psd")

    assert set(filtered[200]) == {"V00001A"}
    assert result[200]["V00002A"]["selections"]["valid-psd"]["status"] == (
        "psd-unavailable"
    )

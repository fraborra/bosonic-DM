# Copyright (C) 2025 Francesco Borra
#

from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pytest

from bosonic_dm.efficiency import (
    build_labels_dicts,
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


def test_build_labels_dicts_can_aggregate_by_detector_group() -> None:
    efficiency_results = {
        200: {
            "V00001A": {
                "expo": 11.0,
                "selections": {
                    "all": {
                        "status": "valid",
                        "efficiency": 1.0,
                        "efficiency_stat_unc": 0.1,
                    }
                },
            },
            "V00002A": {
                "expo": 22.0,
                "selections": {
                    "all": {
                        "status": "valid",
                        "efficiency": 3.0,
                        "efficiency_stat_unc": 0.2,
                    }
                },
            },
        }
    }
    detector_groups = {
        "ICPC group1": {
            "V00001A": {"~p09": "all"},
            "V00002A": {"p09": "all"},
        }
    }
    eres_dict = {
        "p08": {
            "r001": {
                "V00001A": {"usability": "on", "expo": 1.0},
                "V00002A": {"usability": "on", "expo": 2.0},
            }
        },
        "p09": {
            "r001": {
                "V00001A": {"usability": "on", "expo": 10.0},
                "V00002A": {"usability": "on", "expo": 20.0},
            }
        },
    }

    labels = build_labels_dicts(
        efficiency_results,
        group_by="detector_group",
        detector_groups=detector_groups,
        eres_dict=eres_dict,
    )

    group_result = labels["all"][3][200]["ICPC group1"]
    assert group_result["value"] == pytest.approx(61.0 / 21.0)
    assert group_result["exposure"] == pytest.approx(21.0)


def test_build_labels_dicts_requires_group_inputs() -> None:
    with pytest.raises(ValueError, match="detector_groups and eres_dict"):
        build_labels_dicts({}, group_by="detector_group")

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
        "p01": {
            "r001": {
                "V00001A": {
                    "usability": "on",
                    "a": 1.0,
                    "b": 0.0,
                    "a_unc": 0.1,
                    "b_unc": 0.0,
                    "ab_corr": 0.0,
                    "expo": 1.0,
                },
                "V00002A": {
                    "usability": "on",
                    "a": 1.0,
                    "b": 0.0,
                    "a_unc": 0.1,
                    "b_unc": 0.0,
                    "ab_corr": 0.0,
                    "expo": 1.0,
                },
            }
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
    dummy_eres = {"p01": {"r001": {"V00001A": {"usability": "on", "expo": 1.0}}}}
    with pytest.raises(ValueError, match="detector_groups is required"):
        build_labels_dicts({}, eres_dict=dummy_eres, group_by="detector_group")


def test_run_fwhm_only_changes_the_group_selecting_that_run() -> None:
    frame = pl.DataFrame(
        {
            "rawid": [1389, 1389, 8682, 8682],
            "energy": [200.0, 202.0, 200.0, 202.0],
            "sim_e": [200, 200, 200, 200],
            "is_good_channel": [True, True, True, True],
            "has_aoe": [True, True, True, True],
            "is_single_site": [True, True, True, True],
        }
    ).lazy()
    channelmap = {
        "V01389A": SimpleNamespace(daq=SimpleNamespace(rawid=1389)),
        "V08682B": SimpleNamespace(daq=SimpleNamespace(rawid=8682)),
    }
    resolution = {
        "p05": {
            "r001": {
                detector: {
                    "usability": "on",
                    "a": 1.0,
                    "b": 0.0,
                    "a_unc": 0.0,
                    "b_unc": 0.0,
                    "ab_corr": 0.0,
                    "expo": exposure,
                }
                for detector, exposure in {
                    "V01389A": 1.0,
                    "V08682B": 2.0,
                }.items()
            }
        },
        "p09": {
            "r001": {
                detector: {
                    "usability": "on",
                    "a": 9.0,
                    "b": 0.0,
                    "a_unc": 0.0,
                    "b_unc": 0.0,
                    "ab_corr": 0.0,
                    "expo": exposure,
                }
                for detector, exposure in {
                    "V01389A": 10.0,
                    "V08682B": 20.0,
                }.items()
            }
        },
    }

    result = compute_efficiency_from_lazyframe(
        lf=frame,
        eres_dict=resolution,
        simulated_energies=[200],
        chmap=channelmap,
        vertex_counts={200: {"V01389A": 10, "V08682B": 10}},
        half_width_fwhm=1.0,
        selections=["all"],
    )

    assert (
        result[200]["V01389A"]["period_runs"]["p05"]["r001"]["selections"]["all"][
            "n_events"
        ]
        == 1
    )
    assert (
        result[200]["V01389A"]["period_runs"]["p09"]["r001"]["selections"]["all"][
            "n_events"
        ]
        == 2
    )

    detector_groups = {
        "ICPC group1": {
            "V01389A": {"p09": "all"},
            "V08682B": {"p09": "all"},
        },
        "ICPC group2": {
            "V01389A": {"~p09": "all"},
            "V08682B": {"~p09": "all"},
        },
    }
    labels = build_labels_dicts(
        result,
        group_by="detector_group",
        detector_groups=detector_groups,
        eres_dict=resolution,
    )
    groups = labels["all"][3][200]

    assert groups["ICPC group1"]["value"] == pytest.approx(2.5 / 11.0)
    assert groups["ICPC group1"]["exposure"] == pytest.approx(30.0)
    assert groups["ICPC group2"]["value"] == pytest.approx(1.5 / 11.0)
    assert groups["ICPC group2"]["exposure"] == pytest.approx(3.0)

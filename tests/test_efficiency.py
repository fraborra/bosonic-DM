# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from bosonic_dm.efficiency import (
    EFFICIENCY_TABLE_KEYS,
    EFFICIENCY_TABLE_SCHEMA,
    build_labels_dicts,
    combine_run_efficiencies,
    compute_efficiency_from_lazyframe,
    efficiency_table,
    filter_valid_selection_efficiency,
    load_efficiency_table,
    load_primary_counts,
    primary_counts_from_table,
    write_efficiency_table,
)
from bosonic_dm.stats import bayesian_efficiency


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


def _compute(*, lf: pl.LazyFrame, vertex_counts: dict, **kwargs: object) -> dict:
    """Run the efficiency with every event and primary in ``p01``/``r001``."""
    return compute_efficiency_from_lazyframe(
        lf=lf.with_columns(period=pl.lit(1.0), run=pl.lit(1.0)),
        vertex_counts={
            energy: {
                detector: {"p01": {"r001": count}} for detector, count in counts.items()
            }
            for energy, counts in vertex_counts.items()
        },
        **kwargs,
    )


def test_efficiency_marks_missing_primaries_and_psd_unavailable() -> None:
    frame = pl.DataFrame(
        {
            "rawid": [1, 1, 2],
            "energy": [200.0, 201.0, 200.0],
            "sim_e": [200, 200, 200],
            "is_good_channel": [True, True, True],
            "has_aoe": [False, False, True],
            "is_single_site": [False, False, True],
            "coincident_spms": [False, False, False],
        }
    ).lazy()

    result = _compute(
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
    assert first["selections"]["valid-psd"]["exposure_kg_yr"] is None
    assert first["period_runs"]["p01"]["r001"]["selections"]["valid-psd"][
        "exposure_kg_yr"
    ] == pytest.approx(1.0)

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
            "coincident_spms": [False, False],
        }
    ).lazy()

    result = _compute(
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
            "coincident_spms": [False],
        }
    ).lazy()

    result = _compute(
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
            "coincident_spms": [False],
        }
    ).lazy()
    result = _compute(
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
            "coincident_spms": [False, False],
        }
    ).lazy()
    result = _compute(
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


def test_lar_veto_is_applied_to_every_selection() -> None:
    frame = pl.DataFrame(
        {
            "rawid": [1, 1, 1, 1],
            "energy": [200.0, 200.0, 200.0, 200.0],
            "sim_e": [200, 200, 200, 200],
            "is_good_channel": [True, True, True, True],
            "has_aoe": [True, True, True, True],
            "is_single_site": [True, True, False, False],
            "coincident_spms": [False, True, False, True],
        }
    ).lazy()

    with_veto = _compute(
        lf=frame,
        eres_dict=_resolution(),
        simulated_energies=[200],
        chmap=_channelmap(),
        vertex_counts={200: {"V00001A": 10}},
    )
    without_veto = _compute(
        lf=frame,
        eres_dict=_resolution(),
        simulated_energies=[200],
        chmap=_channelmap(),
        vertex_counts={200: {"V00001A": 10}},
        apply_lar_veto=False,
    )

    expected_counts = {
        "all": (2, 4),
        "valid-psd": (2, 4),
        "sse": (1, 2),
        "mse": (1, 2),
    }
    for selection, (with_count, without_count) in expected_counts.items():
        with_result = with_veto[200]["V00001A"]["selections"][selection]
        without_result = without_veto[200]["V00001A"]["selections"][selection]
        assert with_result["n_events"] == with_count
        assert without_result["n_events"] == without_count


def test_efficiency_stores_run_and_aggregate_effective_exposure() -> None:
    frame = pl.DataFrame(
        {
            "rawid": [101, 101],
            "energy": [200.0, 200.0],
            "sim_e": [200, 200],
            "is_good_channel": [True, True],
            "coincident_spms": [False, False],
            "has_aoe": [True, True],
            "is_single_site": [True, True],
        }
    ).lazy()
    resolution = {
        "p01": {
            "r001": {
                "V00001A": {
                    "usability": "on",
                    "a": 1.0,
                    "b": 0.0,
                    "a_unc": 0.0,
                    "b_unc": 0.0,
                    "ab_corr": 0.0,
                    "expo": 3.0,
                }
            }
        }
    }
    channelmap = {"V00001A": SimpleNamespace(daq=SimpleNamespace(rawid=101))}

    result = _compute(
        lf=frame,
        eres_dict=resolution,
        simulated_energies=[200],
        chmap=channelmap,
        vertex_counts={200: {"V00001A": 4}},
        half_width_fwhm=1.0,
        selections=["all"],
    )

    detector = result[200]["V00001A"]
    run_selection = detector["period_runs"]["p01"]["r001"]["selections"]["all"]
    aggregate_selection = detector["selections"]["all"]

    for selection_result in (run_selection, aggregate_selection):
        assert selection_result["exposure_kg_yr"] == pytest.approx(3.0)
        assert selection_result["effective_exposure_kg_yr"] == pytest.approx(
            3.0 * selection_result["efficiency"]
        )
        assert selection_result["effective_exposure_stat_unc_kg_yr"] == pytest.approx(
            3.0 * selection_result["efficiency_stat_unc"]
        )
        assert selection_result["effective_exposure_syst_fwhm_kg_yr"] == pytest.approx(
            3.0 * selection_result["efficiency_syst_fwhm"]
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
            "rawid": [1389, 1389, 8682, 8682] * 2,
            "energy": [200.0, 202.0, 200.0, 202.0] * 2,
            "sim_e": [200] * 8,
            "is_good_channel": [True] * 8,
            "has_aoe": [True] * 8,
            "is_single_site": [True] * 8,
            "coincident_spms": [False] * 8,
            "period": [5.0] * 4 + [9.0] * 4,
            "run": [1.0] * 8,
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
        vertex_counts={
            200: {
                detector: {"p05": {"r001": 10}, "p09": {"r001": 10}}
                for detector in ("V01389A", "V08682B")
            }
        },
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

    # Each group has two runs of ten primaries with exposure weights 1/3 and
    # 2/3: 1 / (1/90 + 4/90) = 18 effective primaries, one prior per group.
    wide_window, _ = bayesian_efficiency(0.2 * 18, 18)
    narrow_window, _ = bayesian_efficiency(0.1 * 18, 18)
    assert groups["ICPC group1"]["value"] == pytest.approx(wide_window)
    assert groups["ICPC group1"]["exposure"] == pytest.approx(30.0)
    assert groups["ICPC group2"]["value"] == pytest.approx(narrow_window)
    assert groups["ICPC group2"]["exposure"] == pytest.approx(3.0)


def _events(
    rawids: list[int],
    energies: list[float],
    sim_e: int = 200,
    period: float = 1.0,
    run: float = 1.0,
) -> dict:
    n = len(rawids)
    return {
        "rawid": rawids,
        "energy": energies,
        "sim_e": [sim_e] * n,
        "is_good_channel": [True] * n,
        "has_aoe": [True] * n,
        "is_single_site": [True] * n,
        "coincident_spms": [False] * n,
        "period": [period] * n,
        "run": [run] * n,
    }


def _four_runs(detectors: dict[str, int]) -> tuple[pl.LazyFrame, dict, dict]:
    """Four runs; each has five events and ten primaries per detector."""
    entry = {
        "usability": "on",
        "a": 1.0,
        "b": 0.0,
        "a_unc": 0.0,
        "b_unc": 0.0,
        "ab_corr": 0.0,
        "expo": 1.0,
    }
    periods = range(1, 5)
    frame = pl.concat(
        pl.DataFrame(_events([rawid] * 5, [200.0] * 5, period=float(period)))
        for period in periods
        for rawid in detectors.values()
    ).lazy()
    resolution = {
        f"p{period:02d}": {"r001": {name: dict(entry) for name in detectors}}
        for period in periods
    }
    counts = {
        200: {
            name: {f"p{period:02d}": {"r001": 10} for period in periods}
            for name in detectors
        }
    }
    return frame, resolution, counts


def test_a_run_uses_only_its_own_events_and_primaries() -> None:
    frame = pl.concat(
        [
            pl.DataFrame(_events([1] * 2, [200.0] * 2, period=1.0)),
            pl.DataFrame(_events([1] * 6, [200.0] * 6, period=2.0)),
        ]
    ).lazy()
    resolution = _resolution()
    resolution["p02"] = {"r001": dict(resolution["p01"]["r001"])}

    result = compute_efficiency_from_lazyframe(
        lf=frame,
        eres_dict=resolution,
        simulated_energies=[200],
        chmap=_channelmap(),
        vertex_counts={200: {"V00001A": {"p01": {"r001": 4}, "p02": {"r001": 8}}}},
        selections=["all"],
    )

    runs = result[200]["V00001A"]["period_runs"]
    first = runs["p01"]["r001"]
    second = runs["p02"]["r001"]
    assert (first["n_primaries"], first["selections"]["all"]["n_events"]) == (4, 2)
    assert (second["n_primaries"], second["selections"]["all"]["n_events"]) == (8, 6)
    assert first["selections"]["all"]["efficiency_mle"] == pytest.approx(0.5)
    assert second["selections"]["all"]["efficiency_mle"] == pytest.approx(0.75)
    detector = result[200]["V00001A"]
    assert detector["n_primaries"] == 12
    assert detector["selections"]["all"]["n_events"] == 8


def test_flat_primary_counts_are_rejected() -> None:
    with pytest.raises(TypeError, match="not split"):
        compute_efficiency_from_lazyframe(
            lf=pl.DataFrame(_events([1], [200.0])).lazy(),
            eres_dict=_resolution(),
            simulated_energies=[200],
            chmap=_channelmap(),
            vertex_counts={200: {"V00001A": 10}},
        )


def test_counts_of_runs_and_detectors_are_combined_before_the_prior() -> None:
    for detectors in ({"V00001A": 1}, {"V00001A": 1, "V00002A": 2}):
        frame, resolution, counts = _four_runs(detectors)
        result = compute_efficiency_from_lazyframe(
            lf=frame,
            eres_dict=resolution,
            simulated_energies=[200],
            chmap=_channelmap(),
            vertex_counts=counts,
            selections=["all"],
        )

        # Four runs of 5 events in 10 primaries: one posterior for 20 in 40.
        detector = result[200]["V00001A"]["selections"]["all"]
        efficiency, stat_unc = bayesian_efficiency(20, 40)
        assert detector["efficiency"] == pytest.approx(efficiency)
        assert detector["efficiency_stat_unc"] == pytest.approx(stat_unc)
        assert detector["n_events"] == 20

        # The group pools the runs of all its detectors in the same way.
        n_runs = 4 * len(detectors)
        group_efficiency, group_unc = bayesian_efficiency(5 * n_runs, 10 * n_runs)
        labels = build_labels_dicts(result, eres_dict={}, group_by="detector_type")
        group = labels["all"][3][200]["ICPC"]
        assert group["value"] == pytest.approx(group_efficiency)
        assert group["unc"] == pytest.approx(group_unc)


def test_many_small_runs_do_not_accumulate_the_prior() -> None:
    # D'Agostini, Sec. 8.11: 1000 measurements of 10 events in 100 trials.
    combined = combine_run_efficiencies([1.0] * 1000, [10] * 1000, [100] * 1000)

    pooled, pooled_unc = bayesian_efficiency(10_000, 100_000)
    assert combined["efficiency"] == pytest.approx(pooled)
    assert combined["efficiency_stat_unc"] == pytest.approx(pooled_unc)
    assert combined["effective_primaries"] == pytest.approx(100_000)
    assert abs(combined["efficiency"] - 0.1) < 0.1 * combined["efficiency_stat_unc"]
    # Averaging the 1000 posterior means keeps one prior per measurement.
    run_mean, _ = bayesian_efficiency(10, 100)
    assert run_mean - 0.1 > 4 * combined["efficiency_stat_unc"]


def test_runs_are_weighted_by_exposure_not_by_primaries() -> None:
    # Run A: exposure 1, 5 events in 10 primaries. Run B: exposure 3, 90 in 100.
    combined = combine_run_efficiencies([1.0, 3.0], [5, 90], [10, 100])

    assert combined["efficiency_mle"] == pytest.approx((1 * 0.5 + 3 * 0.9) / 4)
    # Uneven weights leave fewer effective primaries than the 110 generated.
    assert combined["effective_primaries"] == pytest.approx(16 / (1 / 10 + 9 / 100))
    efficiency, stat_unc = bayesian_efficiency(0.8 * 16 / 0.19, 16 / 0.19)
    assert combined["efficiency"] == pytest.approx(efficiency)
    assert combined["efficiency_stat_unc"] == pytest.approx(stat_unc)


def test_primaries_proportional_to_exposure_give_the_plain_sums() -> None:
    combined = combine_run_efficiencies([2.0, 6.0, 1.0], [3, 40, 9], [20, 60, 10])

    efficiency, stat_unc = bayesian_efficiency(52, 90)
    assert combined["effective_primaries"] == pytest.approx(90)
    assert combined["efficiency_mle"] == pytest.approx(52 / 90)
    assert combined["efficiency"] == pytest.approx(efficiency)
    assert combined["efficiency_stat_unc"] == pytest.approx(stat_unc)


def test_runs_without_exposure_or_primaries_are_left_out() -> None:
    combined = combine_run_efficiencies([1.0, 0.0, 2.0], [5, 7, 0], [10, 10, 0])

    assert combined["efficiency_mle"] == pytest.approx(0.5)
    assert combined["effective_primaries"] == pytest.approx(10)
    assert combine_run_efficiencies([0.0], [1], [10]) is None
    assert combine_run_efficiencies([], [], []) is None


def _sample_result() -> dict:
    """Two energies; V00001A has two runs, V00002A has no primaries."""
    frame = pl.concat(
        [
            pl.DataFrame(_events([1, 1, 1, 2], [200.0, 200.2, 205.0, 200.0])),
            pl.DataFrame(_events([1, 1], [200.0, 200.1], period=2.0, run=3.0)),
            pl.DataFrame(_events([1, 2], [300.0, 300.0], sim_e=300)),
        ]
    ).lazy()
    resolution = _resolution()
    resolution["p02"] = {
        "r003": {"V00001A": {**resolution["p01"]["r001"]["V00001A"], "expo": 3.0}}
    }
    counts = {"V00001A": {"p01": {"r001": 10}, "p02": {"r003": 10}}}
    return compute_efficiency_from_lazyframe(
        lf=frame,
        eres_dict=resolution,
        simulated_energies=[200, 300],
        chmap=_channelmap(),
        vertex_counts={200: counts, 300: counts},
    )


def test_efficiency_table_has_one_row_per_run_selection() -> None:
    result = _sample_result()
    table = efficiency_table(result)

    assert table.schema == EFFICIENCY_TABLE_SCHEMA
    # Per energy: V00001A has 2 runs, V00002A has 1; 4 selections each.
    assert table.height == 2 * (2 + 1) * 4
    assert not table.select(EFFICIENCY_TABLE_KEYS).is_duplicated().any()

    row = table.row(
        by_predicate=(pl.col("sim_e") == 200)
        & (pl.col("detector") == "V00001A")
        & (pl.col("period") == "p02")
        & (pl.col("selection") == "all"),
        named=True,
    )
    nested = result[200]["V00001A"]["period_runs"]["p02"]["r003"]
    assert row["run"] == "r003"
    assert row["status"] == "valid"
    assert row["n_primaries"] == 10
    assert row["n_events"] == nested["selections"]["all"]["n_events"] == 2
    assert row["exposure_kg_yr"] == pytest.approx(3.0)
    assert row["fwhm_keV"] == pytest.approx(nested["fwhm"])
    assert row["efficiency"] == pytest.approx(nested["selections"]["all"]["efficiency"])
    assert row["effective_exposure_kg_yr"] == pytest.approx(3.0 * row["efficiency"])


def test_efficiency_table_keeps_invalid_results_as_nulls() -> None:
    table = efficiency_table(_sample_result()).filter(pl.col("detector") == "V00002A")

    assert set(table["status"]) == {"missing-primaries"}
    assert table["n_primaries"].to_list() == [0] * table.height
    assert table["efficiency"].null_count() == table.height
    assert table["exposure_kg_yr"].null_count() == 0


def test_efficiency_table_round_trip_and_energy_filter(tmp_path: Path) -> None:
    table = efficiency_table(_sample_result())
    path = tmp_path / "dictionaries" / "axio-electric_efficiency.parquet"

    write_efficiency_table(table, path)

    loaded = load_efficiency_table(path)
    assert loaded.equals(table.sort(EFFICIENCY_TABLE_KEYS))
    assert list(path.parent.iterdir()) == [path]
    only_300 = load_efficiency_table(path, energy_range_keV=(250, 300))
    assert set(only_300["sim_e"]) == {300}
    assert load_efficiency_table(path, energy_range_keV=(400, 500)).is_empty()


@pytest.mark.parametrize("group_by", ["detector_type", "detector_group"])
def test_table_and_nested_results_give_the_same_averages(
    group_by: str, tmp_path: Path
) -> None:
    result = _sample_result()
    path = tmp_path / "efficiency.parquet"
    write_efficiency_table(efficiency_table(result), path)
    groups = {"first": {"V00001A": {"p01": "all"}}, "both": ["V00001A", "V00002A"]}

    def averages(efficiency: object) -> dict:
        labels = build_labels_dicts(
            efficiency, eres_dict={}, group_by=group_by, detector_groups=groups
        )
        return {selection: values[3] for selection, values in labels.items()}

    from_nested = averages(result)
    from_table = averages(load_efficiency_table(path))

    assert from_nested["all"][200]
    assert from_table.keys() == from_nested.keys()
    for selection, by_energy in from_nested.items():
        assert from_table[selection].keys() == by_energy.keys()
        for energy, by_group in by_energy.items():
            assert list(from_table[selection][energy]) == list(by_group)
            for group, summary in by_group.items():
                assert from_table[selection][energy][group] == pytest.approx(summary)


def test_efficiency_table_rejects_old_format_and_fractional_energies() -> None:
    old_format = {200: {"V00001A": {"expo": 1.0, "selections": {"all": {}}}}}
    with pytest.raises(ValueError, match="period_runs"):
        efficiency_table(old_format)
    with pytest.raises(ValueError, match="whole keV"):
        efficiency_table({200.5: {}})
    assert efficiency_table({}).schema == EFFICIENCY_TABLE_SCHEMA
    with pytest.raises(ValueError, match="lacks the columns"):
        write_efficiency_table(pl.DataFrame({"sim_e": [200]}), "unused.parquet")


def test_primary_counts_are_read_back_from_the_efficiency_table(
    tmp_path: Path,
) -> None:
    table = efficiency_table(_sample_result())
    # V00002A has a usable run but no counted primaries: stored as zero.
    counts = {
        "V00001A": {"p01": {"r001": 10}, "p02": {"r003": 10}},
        "V00002A": {"p01": {"r001": 0}},
    }

    assert primary_counts_from_table(table) == {200: counts, 300: counts}

    path = tmp_path / "efficiency.parquet"
    write_efficiency_table(table, path)
    assert load_primary_counts(path) == {200: counts, 300: counts}


def test_primaries_are_counted_in_the_run_block_of_their_evtid() -> None:
    from bosonic_dm.geometry import aggregate_vertex_counts_by_run  # noqa: PLC0415

    # Job 0: p03-r000 is evtid 0..4, p03-r001 starts at 5.
    # Job 1: p03-r001 continues from evtid 0, p04-r000 starts at 3.
    ranges = [
        [("p03", "r000", 0, 4), ("p03", "r001", 5, 9)],
        [("p03", "r001", 0, 2), ("p04", "r000", 3, 9)],
    ]
    names = [
        # evtid 1 has two vertices in the same detector: counted once.
        np.array([b"V00001A", b"V00001A", b"V00001A", b"none", b"B00002A"]),
        np.array([b"V00001A", b"V00001A", b"B00002A"]),
    ]
    evtids = [np.array([1, 1, 7, 2, 9]), np.array([2, 3, 8])]

    counts = aggregate_vertex_counts_by_run(names, evtids, ranges)

    assert counts == {
        "B00002A": {"p03": {"r001": 1}, "p04": {"r000": 1}},
        "V00001A": {"p03": {"r000": 1, "r001": 2}, "p04": {"r000": 1}},
    }

# Copyright (C) 2025 Francesco Borra
#

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call

import awkward as ak
import polars as pl
import pytest

import bosonic_dm.background as background_module
from bosonic_dm.background import build_background_dataset
from bosonic_dm.cuts import add_background_cut_flags, pet_to_polars


def _base_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "energy": [100.0, 200.0, 300.0],
            "rawid": [1, 1, 1],
            "detector_name": ["V00001A", "V00001A", "V00001A"],
            "is_good_channel": [True, True, False],
            "coincident_puls": [False, False, False],
            "is_forced": [False, False, False],
            "coincident_muon_offline": [False, False, False],
            "coincident_geds": [True, True, True],
            "is_bb_like": [True, False, True],
            "is_delayed_discharge": [False, False, False],
            "coincident_spms": [False, True, False],
        }
    )


def _pet_events() -> ak.Array:
    return ak.Array(
        {
            "trigger": {
                "timestamp": [1.0, 2.0],
                "is_forced": [False, False],
            },
            "coincident": {
                "muon": [False, False],
                "muon_offline": [False, False],
                "spms": [False, True],
                "spms_experimental": [False, False],
                "puls": [False, False],
                "geds": [True, True],
            },
            "geds": {
                "hit_idx": [[0], [1]],
                "rawid": [[1], [2]],
                "t0": [[0.1], [0.2]],
                "energy": [[100.0], [200.0]],
                "daqenergy": [[101.0], [201.0]],
                "multiplicity": [1, 1],
                "quality": {
                    "is_bb_like": [True, False],
                    "is_good_channel": [[True], [True]],
                    "is_not_bb_like": {
                        "is_delayed_discharge": [False, True],
                    },
                },
                "psd": {
                    "is_good": [[True], [True]],
                    "is_bb_like": [[True], [False]],
                    "drift_time": [[0.3], [0.4]],
                    "low_aoe": {
                        "value": [[0.5], [0.6]],
                        "is_good": [[True], [True]],
                        "is_single_site": [[True], [False]],
                    },
                    "ann": {
                        "value": [[0.7], [0.8]],
                        "is_good": [[True], [True]],
                        "is_single_site": [[True], [False]],
                    },
                },
            },
        }
    )


def test_pet_to_polars_preserves_notebook_cut_inputs() -> None:
    frame = pet_to_polars(
        _pet_events(),
        "p03",
        "r000",
        {1: "V00001A", 2: "V00002A"},
    )

    assert frame["detector_name"].to_list() == ["V00001A", "V00002A"]
    assert frame["coincident_geds"].to_list() == [True, True]
    assert frame["is_delayed_discharge"].to_list() == [False, True]
    assert frame["multiplicity"].to_list() == [1, 1]


def test_cut_flags_preserve_rows_for_notebook_comparisons() -> None:
    result = add_background_cut_flags(
        _base_frame(),
        apply_lar_veto=True,
        comparison_cut_profile="without-bb-like",
    )

    assert result.height == 3
    assert result["passes_baseline"].to_list() == [True, True, False]
    assert result["passes_default"].to_list() == [True, False, False]
    assert result["passes_without_bb_like"].to_list() == [True, True, False]
    assert result["passes_lar"].to_list() == [True, False, True]
    assert result["passes_analysis"].to_list() == [True, False, False]
    assert result["passes_comparison"].to_list() == [True, False, False]


def test_lar_veto_can_be_disabled_without_losing_lar_decision() -> None:
    result = add_background_cut_flags(
        _base_frame(),
        apply_lar_veto=False,
        comparison_cut_profile="without-bb-like",
    )

    assert result["passes_lar"].to_list() == [True, False, True]
    assert result["passes_comparison"].to_list() == [True, True, False]


def test_builder_writes_run_partitions_and_caches_channel_maps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = (
        tmp_path / "pet/l200-p03-r000-phy-tier_pet.lh5",
        tmp_path / "pet/copy-l200-p03-r000-phy-tier_pet.lh5",
        tmp_path / "pet/l200-p04-r001-phy-tier_pet.lh5",
    )
    for source in sources:
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"placeholder")

    context = MagicMock()
    chmap = MagicMock()
    chmap.map.return_value = {1: {"name": "V00001A"}}
    context.get_channelmap_for_run.return_value = chmap
    monkeypatch.setattr(background_module, "read_pet_data", lambda _source: object())
    monkeypatch.setattr(
        background_module,
        "select_multiplicity_one",
        lambda events: events,
    )
    monkeypatch.setattr(
        background_module,
        "pet_to_polars",
        lambda _events, _period, _run, _rawids: _base_frame(),
    )

    output_dir = tmp_path / "parquet/background"
    result = build_background_dataset(
        sources,
        output_dir,
        context,
        apply_lar_veto=True,
        comparison_cut_profile="without-bb-like",
    )

    assert len(result.written_paths) == 3
    assert result.reused_paths == ()
    assert all(path.exists() for path in result.written_paths)
    assert result.written_paths[0].parent == output_dir / "period=p03/run=r000"
    assert result.written_paths[2].parent == output_dir / "period=p04/run=r001"
    assert context.get_channelmap_for_run.call_args_list == [
        call("p03", "r000"),
        call("p04", "r001"),
    ]

    stored = pl.read_parquet(result.written_paths[0])
    assert "passes_analysis" in stored.columns
    assert stored.height == 3

    context.reset_mock()
    cached = build_background_dataset(
        sources,
        output_dir,
        context,
        apply_lar_veto=True,
        comparison_cut_profile="without-bb-like",
    )
    assert cached.written_paths == ()
    assert cached.reused_paths == result.written_paths
    context.get_channelmap_for_run.assert_not_called()


def test_atomic_write_does_not_leave_partial_fragment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "period=p03/run=r000/data.parquet"

    def fail_write(
        _frame: pl.DataFrame,
        _path: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        del args, kwargs
        raise OSError("simulated write failure")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", fail_write)

    with pytest.raises(OSError, match="simulated write failure"):
        background_module._write_parquet_atomic(_base_frame(), destination)

    assert not destination.exists()
    assert list(destination.parent.glob(".*.tmp")) == []

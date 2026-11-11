# Copyright (C) 2026 Francesco Borra
#

"""Readers of the per-job evt-tier simulation files, on small real LH5 files."""

from __future__ import annotations

from pathlib import Path

import lh5
import numpy as np
import polars as pl
import pytest
from lgdo import Array, Table, VectorOfVectors

from bosonic_dm.geometry import aggregate_vertex_counts_by_run, read_run_ranges
from bosonic_dm.io import build_parquet_dataset


def _write_evt_file(
    path: Path,
    *,
    evtid: list[int],
    period: list[int],
    run: list[int],
    energies: list[list[float]],
    spms: list[bool],
) -> Path:
    """Write an evt-tier file with one germanium hit list per event."""
    rawids = [[index + 1 for index in range(len(hits))] for hits in energies]
    flags = [[True] * len(hits) for hits in energies]
    geds = Table(
        col_dict={
            "energy": VectorOfVectors(energies),
            "energy_sum": Array(np.array([float(sum(hits)) for hits in energies])),
            "hit_idx": VectorOfVectors([list(range(len(hits))) for hits in energies]),
            "aoe": VectorOfVectors([[1.0] * len(hits) for hits in energies]),
            "has_aoe": VectorOfVectors(flags),
            "is_good_channel": VectorOfVectors(flags),
            "is_single_site": VectorOfVectors(flags),
            "multiplicity": Array(np.array([len(hits) for hits in energies])),
            "rawid": VectorOfVectors(rawids),
        }
    )
    trigger = Table(
        col_dict={
            "evtid": Array(np.array(evtid)),
            "period": Array(np.array(period)),
            "run": Array(np.array(run)),
        }
    )
    coincident = Table(col_dict={"spms": Array(np.array(spms))})
    path.parent.mkdir(parents=True, exist_ok=True)
    lh5.write(
        Table(col_dict={"geds": geds, "trigger": trigger, "coincident": coincident}),
        "evt",
        str(path),
    )
    return path


@pytest.fixture
def evt_jobs(tmp_path: Path) -> list[Path]:
    """Two jobs of one energy; ``evtid`` restarts in the second job."""
    first = _write_evt_file(
        tmp_path / "evt/job_000-tier_evt.lh5",
        evtid=[3, 7, 12, 20],
        period=[3, 3, 4, 4],
        run=[0, 0, 1, 1],
        energies=[[200.0], [100.0, 50.0], [199.5], [198.0]],
        spms=[False, True, True, False],
    )
    second = _write_evt_file(
        tmp_path / "evt/job_001-tier_evt.lh5",
        evtid=[3, 9],
        period=[4, 5],
        run=[1, 0],
        energies=[[201.0], []],
        spms=[False, False],
    )
    return [first, second]


def test_dataset_is_built_from_the_per_job_evt_files(
    evt_jobs: list[Path], tmp_path: Path
) -> None:
    output_dir = tmp_path / "parquet/axio-electric"

    build_parquet_dataset(
        energies=[200], evt_files={200: evt_jobs}, output_dir=output_dir
    )

    frame = pl.read_parquet(output_dir / "sim_e=200/data.parquet")
    # Multiplicity-one events of both jobs: three of job 0, one of job 1.
    assert frame["energy"].to_list() == [200.0, 199.5, 198.0, 201.0]
    assert frame["evtid"].to_list() == [3, 12, 20, 3]
    assert frame["period"].to_list() == [3, 4, 4, 4]
    assert frame["run"].to_list() == [0, 1, 1, 1]
    assert frame["coincident_spms"].to_list() == [False, True, False, False]
    assert set(frame["sim_e"]) == {200}
    assert {"rawid", "is_good_channel", "has_aoe", "is_single_site"} <= set(
        frame.columns
    )


def test_existing_partition_is_kept_unless_overwrite(
    evt_jobs: list[Path], tmp_path: Path
) -> None:
    output_dir = tmp_path / "parquet"
    build_parquet_dataset(
        energies=[200], evt_files={200: evt_jobs[:1]}, output_dir=output_dir
    )

    with pytest.raises(FileExistsError):
        build_parquet_dataset(
            energies=[200], evt_files={200: evt_jobs}, output_dir=output_dir
        )
    build_parquet_dataset(
        energies=[200], evt_files={200: evt_jobs}, output_dir=output_dir, overwrite=True
    )

    assert pl.read_parquet(output_dir / "sim_e=200/data.parquet").height == 4


def test_run_ranges_are_read_from_every_event_of_a_job(evt_jobs: list[Path]) -> None:
    assert read_run_ranges(evt_jobs[0]) == [
        ("p03", "r000", 3, 7),
        ("p04", "r001", 12, 20),
    ]
    # The second block is known only through an event without germanium hits.
    assert read_run_ranges(evt_jobs[1]) == [
        ("p04", "r001", 3, 3),
        ("p05", "r000", 9, 9),
    ]


def test_overlapping_run_blocks_are_rejected(tmp_path: Path) -> None:
    # As in a merged file: p03-r000 reappears after p04-r001 has started.
    merged = _write_evt_file(
        tmp_path / "merged.lh5",
        evtid=[3, 12, 20],
        period=[3, 4, 3],
        run=[0, 1, 0],
        energies=[[200.0], [200.0], [200.0]],
        spms=[False, False, False],
    )

    with pytest.raises(ValueError, match="overlap"):
        read_run_ranges(merged)


def test_primaries_are_split_by_the_run_blocks_of_the_evt_files(
    evt_jobs: list[Path],
) -> None:
    names = [
        np.array([b"V00001A", b"V00001A", b"V00001A", b"B00002A"]),
        np.array([b"V00001A", b"V00001A"]),
    ]
    # Job 0: evtid 1 is before the first block and is not counted.
    evtids = [np.array([1, 5, 15, 30]), np.array([4, 9])]

    counts = aggregate_vertex_counts_by_run(
        names, evtids, [read_run_ranges(path) for path in evt_jobs]
    )

    assert counts == {
        "B00002A": {"p04": {"r001": 1}},
        "V00001A": {"p03": {"r000": 1}, "p04": {"r001": 2}, "p05": {"r000": 1}},
    }

# Copyright (C) 2025 Francesco Borra
#

"""Run-aware construction of notebook-ready background datasets."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from bosonic_dm.cuts import (
    add_background_cut_flags,
    build_rawid_name_map,
    parse_pet_period_run,
    pet_to_polars,
    read_pet_data,
    select_multiplicity_one,
)

if TYPE_CHECKING:
    from bosonic_dm.pipeline.context import AnalysisContext


@dataclass(frozen=True)
class BackgroundDatasetBuildResult:
    """Parquet fragments written and reused during a dataset build."""

    written_paths: tuple[Path, ...]
    reused_paths: tuple[Path, ...]


def background_partition_path(source: Path, output_dir: Path) -> Path:
    """Return the deterministic Parquet fragment path for one PET source."""
    period, run = parse_pet_period_run(source)
    return output_dir / f"period={period}" / f"run={run}" / f"{source.stem}.parquet"


def _write_parquet_atomic(frame: pl.DataFrame, destination: Path) -> None:
    """Write a Parquet frame atomically within its destination directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
        frame.write_parquet(temporary_path)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_background_dataset(
    pet_files: Sequence[Path],
    output_dir: Path,
    context: AnalysisContext,
    *,
    apply_lar_veto: bool,
    comparison_cut_profile: str,
    overwrite: bool = False,
) -> BackgroundDatasetBuildResult:
    """Convert PET files to run-aware Parquet fragments one file at a time."""
    written_paths: list[Path] = []
    reused_paths: list[Path] = []
    rawid_maps: dict[tuple[str, str], dict[int, str]] = {}

    for source in pet_files:
        period, run = parse_pet_period_run(source)
        destination = background_partition_path(source, output_dir)
        if destination.exists() and not overwrite:
            reused_paths.append(destination)
            continue

        run_key = (period, run)
        if run_key not in rawid_maps:
            chmap = context.get_channelmap_for_run(period, run)
            rawid_maps[run_key] = build_rawid_name_map(chmap)

        events = read_pet_data(source)
        multiplicity_one = select_multiplicity_one(events)
        frame = pet_to_polars(
            multiplicity_one,
            period,
            run,
            rawid_maps[run_key],
        )
        frame = add_background_cut_flags(
            frame,
            apply_lar_veto=apply_lar_veto,
            comparison_cut_profile=comparison_cut_profile,
        )
        _write_parquet_atomic(frame, destination)
        written_paths.append(destination)

    return BackgroundDatasetBuildResult(
        written_paths=tuple(written_paths),
        reused_paths=tuple(reused_paths),
    )

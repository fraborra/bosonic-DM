# Copyright (C) 2026 Francesco Borra
#

"""Run-aware background dataset construction and grouped spectrum preparation."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dbetto
import numpy as np
import polars as pl

from bosonic_dm.cuts import (
    add_background_cut_flags,
    build_rawid_name_map,
    compute_group_exposure,
    filter_dataset,
    parse_pet_period_run,
    pet_to_polars,
    read_pet_data,
    select_multiplicity_one,
    select_usable,
)

if TYPE_CHECKING:
    from bosonic_dm.pipeline.context import AnalysisContext


@dataclass(frozen=True)
class BackgroundDatasetBuildResult:
    """Parquet fragments written and reused during a dataset build."""

    written_paths: tuple[Path, ...]
    reused_paths: tuple[Path, ...]


def _spectrum_bin_edges(
    energy_range_keV: tuple[float, float],
    bin_width_keV: float,
    bin_edges_keV: Sequence[float] | None = None,
) -> np.ndarray:
    """Build validated bin edges, preserving a shorter final bin if necessary."""
    if bin_edges_keV is not None:
        edges = np.asarray(bin_edges_keV, dtype=float).copy()
        if (
            edges.ndim != 1
            or edges.size < 2
            or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)
        ):
            msg = "bin_edges_keV must contain finite, strictly increasing edges."
            raise ValueError(msg)
        return edges

    emin, emax = energy_range_keV
    if not np.isfinite([emin, emax]).all() or emin >= emax:
        msg = "energy_range_keV must have finite, increasing bounds."
        raise ValueError(msg)
    if not np.isfinite(bin_width_keV) or bin_width_keV <= 0:
        msg = "bin_width_keV must be finite and positive."
        raise ValueError(msg)
    edges = np.append(np.arange(emin, emax, bin_width_keV, dtype=float), emax)
    # Avoid a tiny final bin from floating-point rounding.
    if len(edges) > 2 and np.isclose(
        edges[-2],
        emax,
        rtol=0.0,
        atol=8 * np.finfo(float).eps * max(1.0, abs(emin), abs(emax)),
    ):
        edges = np.delete(edges, -2)
    return edges


def spectrum_in_range(
    group: Mapping[str, Any],
    energy_range_keV: tuple[float, float],
) -> dbetto.AttrsDict:
    """Cut a prepared group to complete existing bins inside an energy range.

    Counts, rates, bin widths, and fit masks are sliced without rebinning or
    renormalization. Event energies and selection masks are cut to the same
    retained bin edges. Bounds between bin edges are snapped inward; the
    returned ``bin_edges`` give the actual retained range. Exposure is preserved
    and source arrays are not modified.
    """
    emin, emax = energy_range_keV
    if not np.isfinite([emin, emax]).all() or emin >= emax:
        msg = "energy_range_keV must have finite, increasing bounds."
        raise ValueError(msg)
    source_edges = np.asarray(group["bin_edges"])
    if emin < source_edges[0] or emax > source_edges[-1]:
        msg = "Requested range exceeds the prepared group range."
        raise ValueError(msg)
    bins = np.flatnonzero((source_edges[:-1] >= emin) & (source_edges[1:] <= emax))
    if bins.size == 0:
        msg = "Requested range contains no complete existing bins."
        raise ValueError(msg)
    start, stop = bins[0], bins[-1] + 1
    edges = source_edges[start : stop + 1].copy()
    energy = np.asarray(group["energy"])
    in_range = (energy >= edges[0]) & (energy < edges[-1])
    result = dbetto.AttrsDict(
        {"energy": energy[in_range], "bin_edges": edges, "expo": group["expo"]}
    )
    for field in (
        "bin_centres",
        "bin_widths_keV",
        "fit_mask",
        "counts_analysis",
        "counts_comparison",
        "rate_analysis",
        "rate_comparison",
    ):
        result[field] = np.asarray(group[field])[start:stop].copy()
    for selection in ("analysis", "comparison"):
        result[f"mask_{selection}"] = np.asarray(group[f"mask_{selection}"])[in_range]
    return result


def prepare_background_spectra(
    dataset: pl.LazyFrame | pl.DataFrame,
    groups_dict: Mapping[str, Mapping[str, str | Mapping[str, str | Sequence[str]]]],
    eres_dict: Mapping,
    *,
    energy_range_keV: tuple[float, float] = (60.0, 2000.0),
    bin_width_keV: float = 1.0,
    bin_edges_keV: Sequence[float] | None = None,
    peak_centres_keV: Sequence[float] = (),
    exclusion_width_keV: float = 12.0,
    energy_column: str = "energy",
    analysis_column: str = "passes_analysis",
    comparison_column: str = "passes_comparison",
    exposure_key: str = "expo",
    usable: pl.DataFrame | None = None,
) -> dict[str, dbetto.AttrsDict]:
    """Prepare analysis/comparison counts and rates for each detector group.

    Parameters
    ----------
    dataset
        Background events; a lazy input is filtered and collected once.
    groups_dict
        Named detector selections accepted by ``filter_dataset``.
    eres_dict
        Period/run/detector exposure information for the same data taking as
        ``dataset``, including usable runs with no selected events. Exposures
        must cover both selection columns; cut acceptance is not inferred here.
    energy_range_keV
        Half-open energy range ``[emin, emax)``.
    bin_width_keV
        Positive bin width. The final bin is shorter when necessary.
    bin_edges_keV
        Optional explicit increasing edges, overriding the range and width.
    peak_centres_keV
        Centres of peaks to exclude from fits. Pass
        ``[centre for centre, _, _ in PEAKS]`` for a notebook peak table; an
        empty sequence disables exclusions.
    exclusion_width_keV
        Nonnegative exclusion HALF-width around every peak centre. Bins
        overlapping ``[centre - width, centre + width]`` are flagged in
        ``fit_mask``; events and counts are kept and exposure is unchanged.
    energy_column, analysis_column, comparison_column
        Input energy and boolean selection columns. Null flags count as false.
    exposure_key
        Exposure field used by ``compute_group_exposure``; normally kg yr.
    usable
        Optional ``(period, run, detector_name)`` table of usable detectors
        (see ``build_usability_table``). When given, events outside it are
        dropped before histogramming so counts match the usability-``on``
        exposure. ``None`` keeps every event.

    Returns
    -------
    dict[str, dbetto.AttrsDict]
        One mapping per group with ``energy``, ``mask_analysis``,
        ``mask_comparison``, ``counts_analysis``, ``counts_comparison``,
        ``rate_analysis``, ``rate_comparison``, ``bin_centres``, ``bin_edges``,
        ``bin_widths_keV``, ``expo``, and ``fit_mask``. Rates have units of
        counts / (keV * exposure unit). Event arrays and histograms contain all
        events in range, including those in peak windows. The full histogram
        grid is retained, including empty bins. ``fit_mask`` is false for bins
        overlapping a peak window; pass it to fits (e.g. as Stan's ``use_bin``)
        so those bins are left out of the likelihood. Results support both
        dictionary and attribute access.

    Raises
    ------
    ValueError
        If binning/exclusions are invalid, flags are not boolean, or a group's
        exposure is nonpositive or nonfinite.
    """
    bin_edges = _spectrum_bin_edges(energy_range_keV, bin_width_keV, bin_edges_keV)
    emin, emax = bin_edges[0], bin_edges[-1]

    centres = np.asarray(peak_centres_keV, dtype=float)
    if centres.ndim != 1 or not np.isfinite(centres).all():
        msg = "peak_centres_keV must be a finite one-dimensional sequence."
        raise ValueError(msg)
    if not np.isfinite(exclusion_width_keV) or exclusion_width_keV < 0:
        msg = "exclusion_width_keV must be finite and nonnegative."
        raise ValueError(msg)

    frame = dataset.lazy() if isinstance(dataset, pl.DataFrame) else dataset
    if usable is not None:
        frame = select_usable(frame, usable)
    event_mask = (pl.col(energy_column) >= emin) & (pl.col(energy_column) < emax)
    fit_mask = np.ones(len(bin_edges) - 1, dtype=bool)
    for centre in centres:
        lower, upper = centre - exclusion_width_keV, centre + exclusion_width_keV
        fit_mask &= (bin_edges[1:] <= lower) | (bin_edges[:-1] > upper)
    df = (
        frame.filter(event_mask)
        .select(
            pl.col(energy_column).alias("energy"),
            pl.col(analysis_column).alias("analysis"),
            pl.col(comparison_column).alias("comparison"),
            "detector_name",
            "period",
            "run",
        )
        .collect()
    )
    if any(df.schema[column] != pl.Boolean for column in ("analysis", "comparison")):
        msg = "Analysis and comparison selection columns must be boolean."
        raise ValueError(msg)

    bin_widths = np.diff(bin_edges)
    bin_centres = bin_edges[:-1] + 0.5 * bin_widths
    spectra: dict[str, dbetto.AttrsDict] = {}
    for name, selection in groups_dict.items():
        expo = float(compute_group_exposure(eres_dict, selection, exposure_key))
        if not np.isfinite(expo) or expo <= 0:
            msg = f"Group {name!r} must have a finite positive exposure; got {expo}."
            raise ValueError(msg)
        group = filter_dataset(df, selection)
        energy = group["energy"].to_numpy()
        mask_analysis = group["analysis"].fill_null(False).to_numpy()
        mask_comparison = group["comparison"].fill_null(False).to_numpy()
        counts_analysis, _ = np.histogram(energy[mask_analysis], bins=bin_edges)
        counts_comparison, _ = np.histogram(energy[mask_comparison], bins=bin_edges)
        spectra[name] = dbetto.AttrsDict(
            {
                "energy": energy,
                "mask_analysis": mask_analysis,
                "mask_comparison": mask_comparison,
                "counts_analysis": counts_analysis,
                "counts_comparison": counts_comparison,
                "rate_analysis": counts_analysis / (expo * bin_widths),
                "rate_comparison": counts_comparison / (expo * bin_widths),
                "bin_centres": bin_centres.copy(),
                "bin_edges": bin_edges.copy(),
                "bin_widths_keV": bin_widths.copy(),
                "expo": expo,
                "fit_mask": fit_mask.copy(),
            }
        )
    return spectra


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

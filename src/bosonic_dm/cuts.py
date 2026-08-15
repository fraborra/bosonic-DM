# Copyright (C) 2025 Francesco Borra
#

"""Data loading, quality cuts, and exposure computation for pet-tier data."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import awkward as ak
import lh5
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

PET_DATASET_SCHEMA: dict[str, pl.DataType] = {
    "period": pl.String,
    "run": pl.String,
    "timestamp": pl.Float64,
    "is_forced": pl.Boolean,
    "coincident_muon": pl.Boolean,
    "coincident_muon_offline": pl.Boolean,
    "coincident_spms": pl.Boolean,
    "coincident_spms_experimental": pl.Boolean,
    "coincident_puls": pl.Boolean,
    "coincident_geds": pl.Boolean,
    "hit_idx": pl.Int64,
    "rawid": pl.Int64,
    "t0": pl.Float64,
    "energy": pl.Float64,
    "daqenergy": pl.Float64,
    "multiplicity": pl.Int64,
    "detector_name": pl.String,
    "is_bb_like": pl.Boolean,
    "is_good_channel": pl.Boolean,
    "is_delayed_discharge": pl.Boolean,
    "psd_is_good": pl.Boolean,
    "psd_is_bb_like": pl.Boolean,
    "psd_drift_time": pl.Float64,
    "psd_low_aoe_value": pl.Float64,
    "psd_low_aoe_is_good": pl.Boolean,
    "psd_low_aoe_is_single_site": pl.Boolean,
    "psd_ann_value": pl.Float64,
    "psd_ann_is_good": pl.Boolean,
    "psd_ann_is_single_site": pl.Boolean,
}

DEFAULT_FIELD_MASK: list[str] = [
    "trigger/timestamp",
    "trigger/is_forced",
    "coincident/muon",
    "coincident/muon_offline",
    "coincident/spms",
    "coincident/spms_experimental",
    "coincident/puls",
    "coincident/geds",
    "geds/hit_idx",
    "geds/rawid",
    "geds/t0",
    "geds/psd",
    "geds/energy",
    "geds/daqenergy",
    "geds/multiplicity",
    "geds/quality",
]


def _parse_period_run(filepath: str | Path) -> tuple[str, str]:
    """Extract the period and run identifiers from a pet-tier LH5 filename.

    Parameters
    ----------
    filepath
        Path to an LH5 file. Only the **basename** is parsed.
        Expected format: ``l200-{period}-{run}-phy-tier_pet.lh5``
        Example: ``/some/dir/l200-p03-r000-phy-tier_pet.lh5``

    Returns
    -------
    tuple[str, str]
        ``(period, run)`` — e.g. ``("p03", "r000")``.

    Raises
    ------
    ValueError
        If the filename does not match the expected pattern.
    """
    name = Path(filepath).name
    match = re.search(r"l200-(p\d+)-(r\d+)", name)
    if match is None:
        msg = f"Cannot extract period/run from filename: '{name}'"
        raise ValueError(msg)
    return (match.group(1), match.group(2))


def _build_rawid_name_map(chmap: object) -> dict[int, str]:
    """Build a mapping from DAQ rawid to detector name using the channel map.

    Parameters
    ----------
    chmap
        LEGEND channel-map object (from ``LegendMetadata.channelmap(...)``).
        Must support ``chmap.map("daq.rawid")``.

    Returns
    -------
    dict[int, str]
        Mapping ``{rawid: detector_name}``.
        Example: ``{1104000: "V02160A", 1104001: "B00035A", ...}``.
    """
    rawid_map = chmap.map("daq.rawid")
    result: dict[int, str] = {}
    for rid, info in rawid_map.items():
        try:
            name = info["name"]
            result[int(rid)] = name
        except (KeyError, IndexError, TypeError):
            logger.debug("Skipping rawid %s", rid)
            continue
    return result


def build_rawid_name_map(chmap: object) -> dict[int, str]:
    """Return the DAQ raw-ID to detector-name mapping for a channel map."""
    return _build_rawid_name_map(chmap)


def parse_pet_period_run(filepath: str | Path) -> tuple[str, str]:
    """Return period and run identifiers parsed from a PET filename."""
    return _parse_period_run(filepath)


def read_pet_data(
    lh5_file: str | Path,
    field_mask: Sequence[str] | None = None,
) -> ak.Array:
    """Read a single pet-tier LH5 file, selecting only the specified fields.

    Parameters
    ----------
    lh5_file
        Path to a single ``.lh5`` pet-tier file.
    field_mask
        List of field paths to read (``/``-separated).
        If ``None``, uses ``DEFAULT_FIELD_MASK``.

    Returns
    -------
    ak.Array
        Awkward record array with structure
        ``{trigger: {...}, coincident: {...}, geds: {...}}``.
    """
    mask = list(field_mask) if field_mask is not None else DEFAULT_FIELD_MASK
    return lh5.read_as(
        "evt",
        str(lh5_file),
        field_mask=mask,
        library="ak",
    )


def apply_quality_cuts(evt: ak.Array) -> ak.Array:
    """Apply standard background quality cuts to a pet-tier event array.

    The following cuts are applied as a boolean AND:
      - ``geds.multiplicity == 1`` — single-detector events only
      - ``geds.quality.is_bb_like`` — passes BB-like quality flag
      - ``ak.all(geds.quality.is_good_channel, axis=-1)`` — all hits in good channels
      - ``~coincident.puls`` — no pulser coincidence
      - ``~trigger.is_forced`` — no forced trigger
      - ``~coincident.muon_offline`` — no muon coincidence
      - ``coincident.geds`` — geds in coincidence

    Parameters
    ----------
    evt
        Awkward record array as returned by ``read_pet_data``.

    Returns
    -------
    ak.Array
        Filtered array containing only events passing all cuts.
        May be empty (length 0) if no events pass.
    """
    mask = (
        (evt.geds.multiplicity == 1)
        & (evt.geds.quality.is_bb_like)
        & (ak.all(evt.geds.quality.is_good_channel, axis=-1))
        & (~evt.coincident.puls)
        & (~evt.trigger.is_forced)
        & (~evt.coincident.muon_offline)
        & (evt.coincident.geds)
    )
    return evt[mask]


def select_multiplicity_one(evt: ak.Array) -> ak.Array:
    """Retain events whose HPGe hit fields can be flattened one-to-one."""
    return evt[evt.geds.multiplicity == 1]


def pet_to_polars(
    evt: ak.Array,
    period: str,
    run: str,
    rawid_name_map: Mapping[int, str] | None = None,
) -> pl.DataFrame:
    """Convert a post-cut pet-tier Awkward Array into a flat Polars DataFrame.

    The input array **must** have been filtered to multiplicity == 1
    (e.g. via ``apply_quality_cuts``), so that each nested geds field
    has exactly one value per event and can be flattened with ``ak.flatten``.

    Parameters
    ----------
    evt
        Post-cut Awkward record array.
    period
        Period identifier string (e.g. ``"p03"``).
    run
        Run identifier string (e.g. ``"r000"``).
    rawid_name_map
        Optional mapping ``{rawid: detector_name}``.
        If provided, a ``detector_name`` column is added.
        If ``None``, the column is omitted.

    Returns
    -------
    pl.DataFrame
        Flat DataFrame with one row per event.
    """
    n_events = len(evt)
    if n_events == 0:
        schema = PET_DATASET_SCHEMA.copy()
        if rawid_name_map is None:
            del schema["detector_name"]
        return pl.DataFrame(schema=schema)

    columns: dict[str, np.ndarray | list] = {}
    columns["period"] = [period] * n_events
    columns["run"] = [run] * n_events

    columns["timestamp"] = ak.to_numpy(evt.trigger.timestamp)
    columns["is_forced"] = ak.to_numpy(evt.trigger.is_forced)

    columns["coincident_muon"] = ak.to_numpy(evt.coincident.muon)
    columns["coincident_muon_offline"] = ak.to_numpy(evt.coincident.muon_offline)
    columns["coincident_spms"] = ak.to_numpy(evt.coincident.spms)
    columns["coincident_spms_experimental"] = ak.to_numpy(
        evt.coincident.spms_experimental
    )
    columns["coincident_puls"] = ak.to_numpy(evt.coincident.puls)
    columns["coincident_geds"] = ak.to_numpy(evt.coincident.geds)

    # GEDS fields are nested (one array of hits per event). Since pre-cuts ensure
    # multiplicity == 1, we can safely flatten them to align 1:1 with event-level columns.
    columns["hit_idx"] = ak.to_numpy(ak.flatten(evt.geds.hit_idx))
    columns["rawid"] = ak.to_numpy(ak.flatten(evt.geds.rawid))
    columns["t0"] = ak.to_numpy(ak.flatten(evt.geds.t0))
    columns["energy"] = ak.to_numpy(ak.flatten(evt.geds.energy))
    columns["daqenergy"] = ak.to_numpy(ak.flatten(evt.geds.daqenergy))
    columns["multiplicity"] = ak.to_numpy(evt.geds.multiplicity)

    if rawid_name_map is not None:
        rawid_arr = columns["rawid"]
        columns["detector_name"] = [
            rawid_name_map.get(int(rid), "unknown") for rid in rawid_arr
        ]

    columns["is_bb_like"] = ak.to_numpy(evt.geds.quality.is_bb_like)
    columns["is_good_channel"] = ak.to_numpy(
        ak.flatten(evt.geds.quality.is_good_channel)
    )
    columns["is_delayed_discharge"] = ak.to_numpy(
        evt.geds.quality.is_not_bb_like.is_delayed_discharge
    )

    columns["psd_is_good"] = ak.to_numpy(ak.flatten(evt.geds.psd.is_good))
    columns["psd_is_bb_like"] = ak.to_numpy(ak.flatten(evt.geds.psd.is_bb_like))
    columns["psd_drift_time"] = ak.to_numpy(ak.flatten(evt.geds.psd.drift_time))
    columns["psd_low_aoe_value"] = ak.to_numpy(ak.flatten(evt.geds.psd.low_aoe.value))
    columns["psd_low_aoe_is_good"] = ak.to_numpy(
        ak.flatten(evt.geds.psd.low_aoe.is_good)
    )
    columns["psd_low_aoe_is_single_site"] = ak.to_numpy(
        ak.flatten(evt.geds.psd.low_aoe.is_single_site)
    )
    columns["psd_ann_value"] = ak.to_numpy(ak.flatten(evt.geds.psd.ann.value))
    columns["psd_ann_is_good"] = ak.to_numpy(ak.flatten(evt.geds.psd.ann.is_good))
    columns["psd_ann_is_single_site"] = ak.to_numpy(
        ak.flatten(evt.geds.psd.ann.is_single_site)
    )

    return pl.DataFrame(columns)


def add_background_cut_flags(
    df: pl.DataFrame,
    *,
    apply_lar_veto: bool,
    comparison_cut_profile: str,
) -> pl.DataFrame:
    """Add reusable background cut decisions to a multiplicity-one dataset."""
    profile_columns = {
        "default": "passes_default",
        "without-bb-like": "passes_without_bb_like",
    }
    if comparison_cut_profile not in profile_columns:
        msg = f"Unknown background comparison cut profile: {comparison_cut_profile}"
        raise ValueError(msg)

    baseline = (
        pl.col("is_good_channel")
        & ~pl.col("coincident_puls")
        & ~pl.col("is_forced")
        & ~pl.col("coincident_muon_offline")
        & pl.col("coincident_geds")
    )
    with_profiles = df.with_columns(
        baseline.alias("passes_baseline"),
        (baseline & pl.col("is_bb_like")).alias("passes_default"),
        (baseline & ~pl.col("is_delayed_discharge")).alias("passes_without_bb_like"),
        (~pl.col("coincident_spms")).alias("passes_lar"),
    )
    lar_condition = pl.col("passes_lar") if apply_lar_veto else pl.lit(True)
    return with_profiles.with_columns(
        (pl.col("passes_default") & lar_condition).alias("passes_analysis"),
        (pl.col(profile_columns[comparison_cut_profile]) & lar_condition).alias(
            "passes_comparison"
        ),
    )


def prepare_pet_dataset(
    lh5_files: Sequence[str | Path],
    chmap: object | None = None,
    field_mask: Sequence[str] | None = None,
    cut_func: Callable[[ak.Array], ak.Array] | None = None,
    outfile: str | Path | None = None,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Load, cut, and convert pet-tier LH5 files into a single Polars DataFrame.

    This is the high-level entry point that chains:
      1. ``read_pet_data`` — field selection from one file
      2. ``apply_quality_cuts`` (or custom cuts) — event filtering
      3. ``pet_to_polars`` — awkward → polars with period/run/detector_name

    Parameters
    ----------
    lh5_files
        List of paths to pet-tier ``.lh5`` files.
        Period and run are extracted from each filename.
    chmap
        LEGEND channel-map object. If provided, a ``detector_name``
        column is added by mapping rawid → name.
        If ``None``, the column is omitted.
    field_mask
        Custom field mask. If ``None``, uses ``DEFAULT_FIELD_MASK``.
    cut_func
        Optional custom function to apply cuts. Must take and return an
        Awkward Array. If ``None``, uses ``apply_quality_cuts``.
    outfile
        Optional path where the final Polars DataFrame will be saved
        as a Parquet file. If ``None``, the dataset is not saved to disk.
    overwrite
        If ``True``, overwrite the output file if it already exists.
        If ``False`` and the file exists, raise a ``FileExistsError``.

    Returns
    -------
    pl.DataFrame
        Concatenated DataFrame with all events from all files that pass
        quality cuts. Contains ``period`` and ``run`` columns.
        May be empty if no events pass cuts in any file.
    """
    rawid_name_map: dict[int, str] | None = None
    if chmap is not None:
        rawid_name_map = _build_rawid_name_map(chmap)

    frames: list[pl.DataFrame] = []

    for filepath in lh5_files:
        period, run = _parse_period_run(filepath)
        data = read_pet_data(filepath, field_mask=field_mask)

        if len(data) == 0:
            logger.warning("No events in %s, skipping", filepath)
            continue

        # Dynamically apply cuts: use the provided custom function if available,
        # otherwise fallback to the standard background quality cuts.
        cut_data = cut_func(data) if cut_func is not None else apply_quality_cuts(data)

        if len(cut_data) == 0:
            logger.info("No events pass quality cuts in %s", filepath)
            continue

        df = pet_to_polars(cut_data, period, run, rawid_name_map)
        frames.append(df)

    if not frames:
        logger.warning("No events survived quality cuts in any file")
        df_result = pl.DataFrame()
    else:
        df_result = pl.concat(frames)

    if outfile is not None:
        outpath = Path(outfile)
        if outpath.exists() and not overwrite:
            msg = f"File already exists: {outpath}"
            raise FileExistsError(msg)
        if outpath.parent:
            outpath.parent.mkdir(parents=True, exist_ok=True)
        df_result.write_parquet(outpath)

    return df_result


def filter_dataset(
    df: pl.DataFrame,
    det_selection: Mapping[str, str | Mapping[str, str | Sequence[str]]],
) -> pl.DataFrame:
    """Filter the dataset for specific detectors and period/run combinations.

    Parameters
    ----------
    df
        The pet dataset DataFrame, as returned by ``prepare_pet_dataset``.
    det_selection
        Dictionary specifying the selection per detector.
        Example:
        {
            "V02160A": "all",
            "B00035A": {
                "p03": ["r000", "r004"],
                "p04": "all"
            }
        }

    Returns
    -------
    pl.DataFrame
        A single DataFrame containing all events that match the selection.
    """
    exprs: list[pl.Expr] = []

    for det, det_sel in det_selection.items():
        det_expr = pl.col("detector_name") == det

        if isinstance(det_sel, str) and det_sel.lower() == "all":
            exprs.append(det_expr)
        elif isinstance(det_sel, Mapping):
            # Separate inclusion and exclusion rules
            inc_exprs: list[pl.Expr] = []
            exc_exprs: list[pl.Expr] = []

            for period, runs in det_sel.items():
                is_period_exclusion = period.startswith("~")
                clean_period = period[1:] if is_period_exclusion else period

                # Support for a global "all" period keyword
                if clean_period.lower() == "all":
                    p_expr = pl.lit(True)
                else:
                    p_expr = pl.col("period") == clean_period

                if isinstance(runs, str) and runs.lower() == "all":
                    cond_expr = p_expr
                elif isinstance(runs, Sequence) and not isinstance(runs, str):
                    runs_list = list(runs)
                    is_run_exclusion = all(r.startswith("~") for r in runs_list)
                    is_run_mixed = any(r.startswith("~") for r in runs_list)

                    if is_run_exclusion:
                        clean_runs = [r[1:] for r in runs_list]
                        cond_expr = p_expr & ~pl.col("run").is_in(clean_runs)
                    elif is_run_mixed:
                        msg = f"Cannot mix inclusion and exclusion ('~') for {det}, {period}: {runs}"
                        raise ValueError(msg)
                    else:
                        cond_expr = p_expr & pl.col("run").is_in(runs_list)
                else:
                    msg = f"Invalid runs selection for detector {det}, period {period}: {runs}"
                    raise ValueError(msg)

                if is_period_exclusion:
                    exc_exprs.append(cond_expr)
                else:
                    inc_exprs.append(cond_expr)

            # Combine inclusions (OR). If empty, assume True (include all periods)
            if inc_exprs:
                combined_inc = inc_exprs[0]
                for e in inc_exprs[1:]:
                    combined_inc = combined_inc | e
            else:
                combined_inc = pl.lit(True)

            # Combine exclusions (OR), negate them, and AND with inclusions
            if exc_exprs:
                combined_exc = exc_exprs[0]
                for e in exc_exprs[1:]:
                    combined_exc = combined_exc | e
                final_det_expr = combined_inc & ~combined_exc
            else:
                final_det_expr = combined_inc

            exprs.append(det_expr & final_det_expr)
        else:
            msg = f"Invalid selection for detector {det}: {det_sel}"
            raise ValueError(msg)

    if not exprs:
        return df.clear()

    # Fold all detector-specific expressions into a single, global evaluation tree using OR (|).
    # This allows Polars to optimize and execute the entire filtering logic in a single C++/Rust pass.
    final_expr = exprs[0]
    for e in exprs[1:]:
        final_expr = final_expr | e

    return df.filter(final_expr)


def _matches_selection(
    period: str,
    run: str,
    det_sel: str | Mapping[str, str | Sequence[str]],
) -> bool:
    """Check whether a (period, run) pair is included by a detector selection.

    The selection rules mirror the logic in ``filter_dataset``:

    * ``"all"`` — every period/run is included.
    * A ``dict`` mapping periods to run specifications, where:
      - A period prefixed with ``~`` is an *exclusion* rule.
      - ``"all"`` as the run spec means every run in that period.
      - A list of run strings means only those runs; runs prefixed with
        ``~`` are excluded instead.

    Inclusions are ORed, then ANDed with the negation of ORed exclusions.

    Parameters
    ----------
    period
        Period identifier (e.g. ``"p03"``).
    run
        Run identifier (e.g. ``"r000"``).
    det_sel
        Detector selection, in the same format accepted by
        ``filter_dataset``'s *det_selection* values.

    Returns
    -------
    bool
        ``True`` if the (period, run) pair is included by the selection.
    """
    if isinstance(det_sel, str) and det_sel.lower() == "all":
        return True

    if not isinstance(det_sel, Mapping):
        msg = f"Invalid detector selection: {det_sel}"
        raise ValueError(msg)

    included = False
    excluded = False
    has_inclusion_rule = False

    for sel_period, runs_spec in det_sel.items():
        is_exclusion = sel_period.startswith("~")
        clean_period = sel_period[1:] if is_exclusion else sel_period

        # Track whether any positive (non-exclusion) rule exists,
        # regardless of whether the current period matches.
        if not is_exclusion:
            has_inclusion_rule = True

        # Does this rule match the current period?
        if clean_period.lower() == "all":
            period_matches = True
        else:
            period_matches = period == clean_period

        if not period_matches:
            continue

        # Check run-level matching
        if isinstance(runs_spec, str) and runs_spec.lower() == "all":
            run_matches = True
        elif isinstance(runs_spec, Sequence) and not isinstance(runs_spec, str):
            runs_list = list(runs_spec)
            is_run_exclusion = all(r.startswith("~") for r in runs_list)

            if is_run_exclusion:
                clean_runs = [r[1:] for r in runs_list]
                run_matches = run not in clean_runs
            else:
                run_matches = run in runs_list
        else:
            msg = f"Invalid runs specification: {runs_spec}"
            raise ValueError(msg)

        if run_matches:
            if is_exclusion:
                excluded = True
            else:
                included = True

    # If there are no explicit inclusion rules, default to included
    if not has_inclusion_rule:
        included = True

    return included and not excluded


def matches_period_run_selection(
    period: str,
    run: str,
    detector_selection: str | Mapping[str, str | Sequence[str]],
) -> bool:
    """Return whether a period/run is selected by a detector-group rule."""
    return _matches_selection(period, run, detector_selection)


def compute_group_exposure(
    eres_dict: Mapping,
    group_dict: Mapping[str, str | Mapping[str, str | Sequence[str]]],
    exposure_key: str = "expo",
) -> float:
    """Compute the total exposure for detectors selected by *group_dict*.

    Walks every ``period → run → detector`` entry in *eres_dict*, checks
    whether the detector is requested by *group_dict* (using the same
    inclusion/exclusion rules as ``filter_dataset``), and accumulates the
    requested exposure field for detectors whose ``usability`` is ``"on"``.

    Parameters
    ----------
    eres_dict
        Nested dictionary ``{period: {run: {det_name: {usability, expo, ...}}}}``,
        as loaded from ``eres_dict.yaml``.
    group_dict
        Detector selection dictionary, in the same format accepted by
        ``filter_dataset``'s *det_selection*.  Examples::

            {"V02160A": "all", "V02160B": "all"}          # all periods/runs
            {"V08682B": {"p06": "all", "p07": "all"}}     # specific periods
            {"V01389A": {"~p09": "all"}}                   # exclude p09
    exposure_key
        Key containing the exposure in each detector's run information.

    Returns
    -------
    float
        Total exposure in the same units as the *exposure_key* values in
        *eres_dict* (typically kg·yr).

    Examples
    --------
    >>> eres = {"p03": {"r000": {"V02160A": {"usability": "on", "expo": 0.025}}}}
    >>> compute_group_exposure(eres, {"V02160A": "all"})
    0.025
    """
    total_expo = 0.0

    for period, runs in eres_dict.items():
        for run, detectors in runs.items():
            for det_name, det_info in detectors.items():
                # Skip detectors not in the group
                if det_name not in group_dict:
                    continue

                # Skip detectors that are not usable
                if det_info.get("usability") != "on":
                    continue

                # Check period/run inclusion using the same rules as filter_dataset
                det_sel = group_dict[det_name]
                if matches_period_run_selection(period, run, det_sel):
                    total_expo += det_info[exposure_key]

    return total_expo

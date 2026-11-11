# Copyright (C) 2026 Francesco Borra
#

"""Detection efficiency computation for bosonic-DM analysis."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from bosonic_dm.constants import DET_TYPE_MAP
from bosonic_dm.cuts import matches_period_run_selection
from bosonic_dm.io import (
    get_mean_fcc_det_group,
    get_mean_fcc_det_type,
    write_parquet_atomic,
)
from bosonic_dm.resolution import compute_fwhm, propagate_resolution_uncertainty
from bosonic_dm.stats import bayesian_efficiency
from bosonic_dm.utils import period_run_number

logger = logging.getLogger(__name__)

_SELECTION_REQUIREMENTS = {
    "all": {},
    "valid-psd": {"has_aoe": True},
    "sse": {"has_aoe": True, "is_single_site": True},
    "mse": {"has_aoe": True, "is_single_site": False},
}

# Float results of one run-level selection, as named in the nested output of
# ``compute_efficiency_from_lazyframe`` and in the efficiency table.
_SELECTION_RESULT_FIELDS = (
    "efficiency_mle",
    "efficiency",
    "efficiency_stat_unc",
    "efficiency_syst_fwhm",
    "effective_exposure_kg_yr",
    "effective_exposure_stat_unc_kg_yr",
    "effective_exposure_syst_fwhm_kg_yr",
)

# Columns that identify one row of the efficiency table.
EFFICIENCY_TABLE_KEYS = ("sim_e", "detector", "period", "run", "selection")

# Efficiency table: one row per simulated energy, detector, period, run, and
# selection. ``exposure_kg_yr``, ``fwhm_keV`` and ``fwhm_unc_keV`` describe the
# run, as does ``n_primaries`` (primaries assigned to that run in that
# detector); ``psd_available`` describes the detector.
EFFICIENCY_TABLE_SCHEMA: dict[str, type[pl.DataType]] = {
    "sim_e": pl.Int64,
    "detector": pl.String,
    "period": pl.String,
    "run": pl.String,
    "selection": pl.String,
    "status": pl.String,
    "psd_available": pl.Boolean,
    "n_primaries": pl.Int64,
    "n_events": pl.Int64,
    "exposure_kg_yr": pl.Float64,
    "fwhm_keV": pl.Float64,
    "fwhm_unc_keV": pl.Float64,
    **dict.fromkeys(_SELECTION_RESULT_FIELDS, pl.Float64),
}


def build_selection_metadata(
    selections: Sequence[str],
    *,
    half_width_fwhm: float,
    apply_lar_veto: bool,
) -> dict[str, object]:
    """Describe the event selections used to produce an efficiency table."""
    unknown_selections = set(selections) - set(_SELECTION_REQUIREMENTS)
    if unknown_selections:
        msg = f"Unknown efficiency selections: {sorted(unknown_selections)}"
        raise ValueError(msg)

    return {
        "energy_window": {
            "reconstructed_energy_column": "energy",
            "center_energy_column": "sim_e",
            "half_width_fwhm": half_width_fwhm,
            "resolution_scope": "period-run-detector",
        },
        "good_channel": {
            "column": "is_good_channel",
            "pass_value": True,
        },
        "lar_veto": {
            "applied": apply_lar_veto,
            "column": "coincident_spms",
            "pass_value": False,
            "null_policy": "reject",
        },
        "selections": {
            selection: {"event_requirements": dict(_SELECTION_REQUIREMENTS[selection])}
            for selection in selections
        },
    }


def _selection_result(
    *,
    n_events: int,
    n_events_up: int,
    n_events_down: int,
    n_primaries: int,
    exposure_kg_yr: float,
    psd_available: bool | None,
    selection: str,
) -> dict[str, object]:
    """Build one run-level selection result with defined edge-case status."""
    result: dict[str, object] = {
        "status": "valid",
        "n_events": n_events,
        "efficiency_mle": None,
        "efficiency": None,
        "efficiency_stat_unc": None,
        "efficiency_syst_fwhm": None,
        "exposure_kg_yr": exposure_kg_yr,
        "effective_exposure_kg_yr": None,
        "effective_exposure_stat_unc_kg_yr": None,
        "effective_exposure_syst_fwhm_kg_yr": None,
    }
    if n_primaries <= 0:
        result["status"] = "missing-primaries"
        return result
    if selection != "all" and psd_available is False:
        result["status"] = "psd-unavailable"
        return result
    if any(
        count < 0 or count > n_primaries
        for count in (n_events, n_events_up, n_events_down)
    ):
        result["status"] = "invalid-counts"
        return result

    efficiency, stat_unc = bayesian_efficiency(n_events, n_primaries)
    efficiency_up, _ = bayesian_efficiency(n_events_up, n_primaries)
    efficiency_down, _ = bayesian_efficiency(n_events_down, n_primaries)
    result.update(
        {
            "efficiency_mle": float(n_events / n_primaries),
            "efficiency": efficiency,
            "efficiency_stat_unc": stat_unc,
            "efficiency_syst_fwhm": abs(efficiency_up - efficiency_down) / 2.0,
            "effective_exposure_kg_yr": exposure_kg_yr * efficiency,
            "effective_exposure_stat_unc_kg_yr": exposure_kg_yr * stat_unc,
            "effective_exposure_syst_fwhm_kg_yr": exposure_kg_yr
            * abs(efficiency_up - efficiency_down)
            / 2.0,
        }
    )
    return result


def combine_run_efficiencies(
    exposures: Sequence[float] | np.ndarray,
    n_events: Sequence[float] | np.ndarray,
    n_primaries: Sequence[float] | np.ndarray,
) -> dict[str, float] | None:
    """Combine the counts of several runs into one exposure-weighted efficiency.

    Every run has its own events and primaries. Its counts are weighted by
    its exposure divided by its primaries, so the weighted primaries of a run
    equal its exposure, and the ratio of the weighted sums is the
    exposure-weighted mean of the run-level ``n_events / n_primaries``.

    The Jeffreys prior is applied once, to the combined counts. Averaging
    run-level posterior means instead would put in one prior per run and pull
    the result towards 0.5 by an amount that does not shrink when runs are
    added (D'Agostini, *Bayesian Reasoning in Data Analysis*, Sec. 8.11).

    Weighted counts are not integers, so the posterior uses the effective
    number of primaries ``n_eff = (sum X)^2 / sum(X^2 / n)``, which gives the
    binomial variance of the weighted mean, and ``n_eff`` times that mean as
    effective events. When primaries are proportional to exposure these are
    the plain sums of primaries and events, and the result is exact.

    Parameters
    ----------
    exposures
        Exposure of every run entry; entries without a positive exposure are
        left out.
    n_events, n_primaries
        Selected events and generated primaries of every run entry; entries
        without primaries are left out.

    Returns
    -------
    dict[str, float] | None
        ``efficiency_mle`` (weighted mean without a prior), ``efficiency``
        and ``efficiency_stat_unc`` (posterior mean and standard deviation),
        and ``effective_primaries``. ``None`` if no entry can be used.
    """
    exposure = np.asarray(exposures, dtype=float)
    events = np.asarray(n_events, dtype=float)
    primaries = np.asarray(n_primaries, dtype=float)
    usable = (
        np.isfinite(exposure)
        & np.isfinite(events)
        & np.isfinite(primaries)
        & (exposure > 0)
        & (primaries > 0)
    )
    if not usable.any():
        return None
    exposure, events, primaries = exposure[usable], events[usable], primaries[usable]
    weights = exposure / np.sum(exposure)
    # The clip only removes rounding: every run has events <= primaries.
    efficiency_mle = float(np.clip(np.sum(weights * events / primaries), 0.0, 1.0))
    effective_primaries = float(1.0 / np.sum(weights**2 / primaries))
    efficiency, stat_unc = bayesian_efficiency(
        efficiency_mle * effective_primaries, effective_primaries
    )
    return {
        "efficiency_mle": efficiency_mle,
        "efficiency": float(efficiency),
        "efficiency_stat_unc": float(stat_unc),
        "effective_primaries": effective_primaries,
    }


def _aggregate_run_selection(
    run_results: Sequence[Mapping[str, object]],
    selection: str,
) -> dict[str, object]:
    """Combine one selection of a detector after retaining its run-level results.

    All run results belong to one detector and one simulated energy. The
    efficiency comes from ``combine_run_efficiencies``; the FWHM systematic is
    the exposure-weighted mean of the run-level values.
    """
    valid: list[tuple[float, float, Mapping[str, object]]] = []
    statuses: list[str] = []
    for run_result in run_results:
        run_selections = run_result["selections"]
        if not isinstance(run_selections, Mapping):
            msg = "Run result selections must be a mapping."
            raise TypeError(msg)
        selection_result = run_selections[selection]
        if not isinstance(selection_result, Mapping):
            msg = "Run selection result must be a mapping."
            raise TypeError(msg)
        status = str(selection_result["status"])
        statuses.append(status)
        exposure = float(run_result["expo"])
        if status == "valid" and exposure > 0:
            valid.append((exposure, float(run_result["n_primaries"]), selection_result))

    combined = combine_run_efficiencies(
        [item[0] for item in valid],
        [float(item[2]["n_events"]) for item in valid],
        [item[1] for item in valid],
    )
    if combined is None:
        if "missing-primaries" in statuses:
            status = "missing-primaries"
        elif "invalid-counts" in statuses:
            status = "invalid-counts"
        elif "psd-unavailable" in statuses:
            status = "psd-unavailable"
        else:
            status = "missing-exposure"
        return {
            "status": status,
            "n_events": None,
            "efficiency_mle": None,
            "efficiency": None,
            "efficiency_stat_unc": None,
            "efficiency_syst_fwhm": None,
            "exposure_kg_yr": None,
            "effective_exposure_kg_yr": None,
            "effective_exposure_stat_unc_kg_yr": None,
            "effective_exposure_syst_fwhm_kg_yr": None,
        }

    weights = np.asarray([item[0] for item in valid], dtype=float)
    exposure_kg_yr = float(np.sum(weights))
    efficiency_mle = combined["efficiency_mle"]
    efficiency = combined["efficiency"]
    efficiency_stat_unc = combined["efficiency_stat_unc"]
    efficiency_syst_fwhm = float(
        np.sum(
            weights
            / exposure_kg_yr
            * np.asarray([float(item[2]["efficiency_syst_fwhm"]) for item in valid])
        )
    )
    return {
        "status": "valid",
        "n_events": int(sum(int(item[2]["n_events"]) for item in valid)),
        "efficiency_mle": efficiency_mle,
        "efficiency": efficiency,
        "efficiency_stat_unc": efficiency_stat_unc,
        "efficiency_syst_fwhm": efficiency_syst_fwhm,
        "exposure_kg_yr": exposure_kg_yr,
        "effective_exposure_kg_yr": exposure_kg_yr * efficiency,
        "effective_exposure_stat_unc_kg_yr": exposure_kg_yr * efficiency_stat_unc,
        "effective_exposure_syst_fwhm_kg_yr": exposure_kg_yr * efficiency_syst_fwhm,
    }


def compute_efficiency_from_lazyframe(
    lf: pl.LazyFrame,
    eres_dict: dict,
    simulated_energies: Sequence[int],
    chmap: object,
    vertex_counts: Mapping[int, Mapping[str, Mapping[str, Mapping[str, int]]]],
    half_width_fwhm: float = 2.0,
    selections: Sequence[str] = ("all", "valid-psd", "sse", "mse"),
    apply_lar_veto: bool = True,
) -> dict:
    """Compute run-aware per-detector efficiencies for all selections.

    The simulation assigns every primary, and so every event, to one
    period/run. For each usable period/run/detector entry in *eres_dict* the
    efficiency is the number of that run's good-channel events inside the FEP
    window set by that run's resolution, divided by the primaries generated in
    the detector for that run. Run-level results are retained under
    ``period_runs``. The detector-level ``selections`` fields combine the
    run-level counts with exposure weights and apply the prior once
    (``combine_run_efficiencies``).

    When *apply_lar_veto* is true, every selection additionally requires no
    SiPM coincidence:
    - all: events in FEP window
    - valid-psd: events in FEP window with has_aoe == True
    - sse: valid-psd with is_single_site == True
    - mse: valid-psd with is_single_site == False

    Parameters
    ----------
    lf
        A Polars *lazy* scan of the parquet dataset partitioned by ``sim_e``.
        Expected columns: ``rawid``, ``energy``, ``sim_e``, ``is_good_channel``,
        ``has_aoe``, ``is_single_site``, and the numeric ``period`` and ``run``
        each event was assigned to. When *apply_lar_veto* is true,
        ``coincident_spms`` is also required.
    eres_dict
        Nested dictionary with structure
        ``{period: {run: {detector_name: {"usability", "expo", "a", "b",
        "a_unc", "b_unc", "ab_corr"}}}}``.
    simulated_energies
        List of simulated energies (in keV) to iterate over.
    chmap
        LEGEND channel-map object (``LegendMetadata.channelmap(...)``).
    vertex_counts
        Primaries generated in each detector, split by the period and run the
        simulation assigned them to:
        ``{energy: {detector_name: {period: {run: n_primaries}}}}``
        (``geometry.aggregate_vertex_counts_by_run``).
    half_width_fwhm
        The multiplier for the FWHM to define the integration window. Defaults to 2.0.
    selections
        Selection names to compute. Supported values are ``all``,
        ``valid-psd``, ``sse``, and ``mse``.
    apply_lar_veto
        If true, require ``coincident_spms == False`` for every selection.
        Null SiPM-coincidence decisions are rejected conservatively.

    Returns
    -------
    dict
        Nested dictionary with structure::

            {
                energy: {
                    detector_name: {
                        "n_primaries":      int,
                        "expo":             float,
                        "period_runs": {
                            period: {run: {"n_primaries", "expo", "fwhm", "selections"}}
                        },
                        "selections": {
                            "all": {
                                "n_events": int,
                                "efficiency_mle": float,
                                "efficiency": float,
                                "efficiency_stat_unc": float,
                                "efficiency_syst_fwhm": float,
                                "exposure_kg_yr": float,
                                "effective_exposure_kg_yr": float,
                                "effective_exposure_stat_unc_kg_yr": float,
                                "effective_exposure_syst_fwhm_kg_yr": float,
                            },
                            "valid-psd": { ... },
                            "sse": { ... },
                            "mse": { ... }
                        }
                    },
                    ...
                },
                ...
            }
    """
    unknown_selections = set(selections) - set(_SELECTION_REQUIREMENTS)
    if unknown_selections:
        msg = f"Unknown efficiency selections: {sorted(unknown_selections)}"
        raise ValueError(msg)

    ratio_dict: dict = {}

    for ene in tqdm(simulated_energies):
        ene_key = int(ene)
        ratio_dict[ene_key] = {}

        resolution_entries: dict[str, list[dict[str, object]]] = {}
        for period, runs in eres_dict.items():
            for run, detectors in runs.items():
                for det_name, values in detectors.items():
                    if values.get("usability") != "on":
                        continue
                    if not all(key in values for key in ("a", "b", "expo")):
                        logger.info(
                            "Missing resolution values for %s in %s-%s; skipping",
                            det_name,
                            period,
                            run,
                        )
                        continue
                    fwhm = float(compute_fwhm(values["a"], values["b"], ene))
                    fwhm_unc = float(
                        propagate_resolution_uncertainty(
                            values["a"],
                            values["b"],
                            values.get("a_unc", 0.0),
                            values.get("b_unc", 0.0),
                            values.get("ab_corr", 0.0),
                            ene,
                        )
                    )
                    if not np.isfinite(fwhm) or fwhm <= 0:
                        logger.warning(
                            "Invalid FWHM for %s in %s-%s at %d keV; skipping",
                            det_name,
                            period,
                            run,
                            ene_key,
                        )
                        continue
                    resolution_entries.setdefault(det_name, []).append(
                        {
                            "period": str(period),
                            "run": str(run),
                            "expo": float(values["expo"]),
                            "fwhm": fwhm,
                            "fwhm_unc": fwhm_unc,
                        }
                    )

        if not resolution_entries:
            logger.warning(
                "No valid resolution info found in eres_dict, skipping energy %d keV",
                ene_key,
            )
            continue
        vtx_counts_for_ene = vertex_counts.get(ene_key, {})

        rawids: dict[str, int] = {}
        for det_name in resolution_entries:
            try:
                rawids[det_name] = int(chmap[det_name].daq.rawid)
            except (KeyError, AttributeError):
                logger.warning("Cannot resolve rawid for %s, skipping", det_name)

        if not rawids:
            continue

        # The simulation assigns every primary to one period/run. A run's
        # efficiency uses only its own events and its own primaries.
        base_filter = (
            pl.col("is_good_channel")
            & (pl.col("sim_e") == ene)
            & pl.col("rawid").is_in(list(rawids.values()))
        )
        if apply_lar_veto:
            base_filter &= pl.col("coincident_spms").eq(False).fill_null(False)

        df_ene = (
            lf.filter(base_filter)
            .select("rawid", "energy", "has_aoe", "is_single_site", "period", "run")
            .collect()
        )

        # Fill nulls in booleans with False to allow safe aggregation
        df_ene = df_ene.with_columns(
            pl.col("has_aoe").fill_null(False),
            pl.col("is_single_site").fill_null(False),
            pl.col("period").cast(pl.Int64, strict=False).fill_null(-1),
            pl.col("run").cast(pl.Int64, strict=False).fill_null(-1),
        )

        for det_name, rawid in rawids.items():
            detector_frame = df_ene.filter(pl.col("rawid") == rawid)
            energies = detector_frame["energy"].to_numpy()
            distances_from_peak = np.abs(energies - ene)
            has_aoe = detector_frame["has_aoe"].to_numpy().astype(bool)
            is_single_site = detector_frame["is_single_site"].to_numpy().astype(bool)
            event_period = detector_frame["period"].to_numpy()
            event_run = detector_frame["run"].to_numpy()
            psd_available = bool(np.any(has_aoe)) if len(detector_frame) else None
            selection_masks = {
                "all": np.ones(len(detector_frame), dtype=bool),
                "valid-psd": has_aoe,
                "sse": has_aoe & is_single_site,
                "mse": has_aoe & ~is_single_site,
            }
            detector_counts = vtx_counts_for_ene.get(det_name, {})
            if not isinstance(detector_counts, Mapping):
                msg = (
                    f"Primary counts for {det_name} at {ene_key} keV are not split "
                    "by period and run; rerun the efficiencies stage with "
                    "overwrite enabled."
                )
                raise TypeError(msg)
            period_runs: dict[str, dict[str, dict[str, object]]] = {}
            flat_run_results: list[dict[str, object]] = []

            for resolution_entry in resolution_entries[det_name]:
                period = str(resolution_entry["period"])
                run = str(resolution_entry["run"])
                fwhm = float(resolution_entry["fwhm"])
                fwhm_unc = max(float(resolution_entry["fwhm_unc"]), 0.0)
                n_primaries = int(detector_counts.get(period, {}).get(run, 0))
                in_run = (event_period == period_run_number(period)) & (
                    event_run == period_run_number(run)
                )
                windows = (
                    fwhm,
                    fwhm + fwhm_unc,
                    max(fwhm - fwhm_unc, 0.0),
                )
                run_selections: dict[str, dict[str, object]] = {}
                for selection in selections:
                    counts: list[int] = []
                    for window_fwhm in windows:
                        in_window = distances_from_peak <= (
                            half_width_fwhm * window_fwhm
                        )
                        counts.append(
                            int(
                                np.count_nonzero(
                                    in_run & in_window & selection_masks[selection]
                                )
                            )
                        )
                    run_selections[selection] = _selection_result(
                        n_events=counts[0],
                        n_events_up=counts[1],
                        n_events_down=counts[2],
                        n_primaries=n_primaries,
                        exposure_kg_yr=float(resolution_entry["expo"]),
                        psd_available=psd_available,
                        selection=selection,
                    )

                run_result: dict[str, object] = {
                    "status": "valid" if n_primaries > 0 else "missing-primaries",
                    "n_primaries": n_primaries,
                    "expo": float(resolution_entry["expo"]),
                    "fwhm": fwhm,
                    "fwhm_unc": fwhm_unc,
                    "selections": run_selections,
                }
                period_runs.setdefault(period, {})[run] = run_result
                flat_run_results.append(run_result)

            n_primaries = sum(int(item["n_primaries"]) for item in flat_run_results)
            det_out = {
                "status": "valid" if n_primaries > 0 else "missing-primaries",
                "n_primaries": n_primaries,
                "expo": float(sum(float(item["expo"]) for item in flat_run_results)),
                "psd_available": psd_available,
                "period_runs": period_runs,
                "selections": {
                    selection: _aggregate_run_selection(flat_run_results, selection)
                    for selection in selections
                },
            }
            ratio_dict[ene_key][det_name] = det_out

    return ratio_dict


def filter_non_zero_efficiency(ratio_dict: Mapping) -> dict:
    """Filter ratio_dict to keep only entries where ratio (efficiency) is not zero.

    Parameters
    ----------
    ratio_dict
        Nested dictionary of efficiencies, structured as:
        ``{energy: {detector_name: {"ratio": float, ...}, ...}, ...}``.

    Returns
    -------
    dict
        A new nested dictionary containing only the detectors (and energies)
        with a non-zero ratio.
    """
    filtered: dict = {}
    for ene, det_dict in ratio_dict.items():
        filtered_dets = {}
        for det_name, info in det_dict.items():
            if info.get("ratio", 0.0) != 0.0:
                filtered_dets[det_name] = info
        if filtered_dets:
            filtered[ene] = filtered_dets
    return filtered


def filter_valid_selection_efficiency(
    ratio_dict: Mapping,
    selection: str,
) -> dict:
    """Keep only detector entries with a valid result for ``selection``.

    This excludes unavailable PSD detectors by status without confusing
    unavailability with a measured zero efficiency.
    """
    filtered: dict = {}
    for energy, detector_results in ratio_dict.items():
        valid_detectors = {
            detector: result
            for detector, result in detector_results.items()
            if result.get("selections", {}).get(selection, {}).get("status") == "valid"
        }
        if valid_detectors:
            filtered[energy] = valid_detectors
    return filtered


def restructure_efficiency_by_selection(ratio_dict: Mapping) -> dict:
    """Restructure an efficiency dictionary from `energy -> detector -> selection` to `selection -> energy -> detector`.

    Parameters
    ----------
    ratio_dict
        Nested dictionary of efficiencies, structured as:
        ``{energy: {detector_name: {"selections": {"selection_name": dict, ...}, ...}, ...}``.

    Returns
    -------
    dict
        A restructured nested dictionary structured as:
        ``{selection: {energy: {detector_name: dict, ...}, ...}``.
    """
    restructured: dict = {}
    for energy, det_dict in ratio_dict.items():
        for det_name, det_info in det_dict.items():
            # Handle standard new format where selections are grouped under "selections"
            if "selections" in det_info:
                # Extract top-level info (excluding "selections")
                base_info = {k: v for k, v in det_info.items() if k != "selections"}

                selections = det_info["selections"]
                for selection, sel_data in selections.items():
                    # Merge top-level info with the selection-specific info
                    merged_data = base_info.copy()
                    merged_data.update(sel_data)

                    restructured.setdefault(selection, {}).setdefault(energy, {})[
                        det_name
                    ] = merged_data
            else:
                # Fallback for simple/flat dictionaries (e.g., if there are no selections)
                restructured.setdefault("all", {}).setdefault(energy, {})[det_name] = (
                    det_info
                )

    return restructured


def _integer_energy(energy: object) -> int:
    """Return a simulated energy as an integer number of keV."""
    value = float(energy)
    if not value.is_integer():
        msg = f"Simulated energies must be whole keV values; got {energy!r}."
        raise ValueError(msg)
    return int(value)


def _optional(value: object, cast: type) -> object:
    """Cast a value that may be missing."""
    return None if value is None else cast(value)


def efficiency_table(efficiency_dict: Mapping) -> pl.DataFrame:
    """Flatten run-level efficiency results into one row per selection.

    Parameters
    ----------
    efficiency_dict
        Output of ``compute_efficiency_from_lazyframe``:
        ``{energy: {detector: {"period_runs": {period: {run: {...}}}}}}``.

    Returns
    -------
    pl.DataFrame
        Table with the columns of ``EFFICIENCY_TABLE_SCHEMA`` and one row per
        simulated energy, detector, period, run, and selection. Fields a
        result does not define are null, e.g. the efficiency of a selection
        whose ``status`` is not ``"valid"``. Detector and group averages are
        not stored; compute them with ``aggregate_run_aware_efficiencies``.

    Raises
    ------
    ValueError
        If a detector result has no run-level ``period_runs`` entry
        (old-format input), or an energy is not a whole number of keV.
    """
    frames: list[pl.DataFrame] = []
    for energy, detector_results in efficiency_dict.items():
        sim_e = _integer_energy(energy)
        columns: dict[str, list[object]] = {
            name: [] for name in EFFICIENCY_TABLE_SCHEMA
        }
        for detector, detector_result in detector_results.items():
            if "period_runs" not in detector_result:
                msg = (
                    f"Detector {detector!r} at {sim_e} keV has no run-level "
                    "'period_runs' entry; regenerate the efficiencies with the "
                    "current efficiency stage."
                )
                raise ValueError(msg)
            psd_available = _optional(detector_result.get("psd_available"), bool)
            for period, runs in detector_result["period_runs"].items():
                for run, run_result in runs.items():
                    for selection, result in run_result.get("selections", {}).items():
                        columns["sim_e"].append(sim_e)
                        columns["detector"].append(str(detector))
                        columns["period"].append(str(period))
                        columns["run"].append(str(run))
                        columns["selection"].append(str(selection))
                        columns["status"].append(_optional(result.get("status"), str))
                        columns["psd_available"].append(psd_available)
                        columns["n_primaries"].append(
                            _optional(run_result.get("n_primaries"), int)
                        )
                        columns["n_events"].append(
                            _optional(result.get("n_events"), int)
                        )
                        columns["exposure_kg_yr"].append(
                            _optional(run_result.get("expo"), float)
                        )
                        columns["fwhm_keV"].append(
                            _optional(run_result.get("fwhm"), float)
                        )
                        columns["fwhm_unc_keV"].append(
                            _optional(run_result.get("fwhm_unc"), float)
                        )
                        for field in _SELECTION_RESULT_FIELDS:
                            columns[field].append(_optional(result.get(field), float))
        # One frame per energy keeps the temporary Python lists small.
        frames.append(pl.DataFrame(columns, schema=EFFICIENCY_TABLE_SCHEMA))
    if not frames:
        return pl.DataFrame(schema=EFFICIENCY_TABLE_SCHEMA)
    return pl.concat(frames)


def efficiency_table_path(dictionaries_root: str | Path, interaction: str) -> Path:
    """Return where the pipeline stores the efficiency table of an interaction."""
    return Path(dictionaries_root) / f"{interaction}_efficiency.parquet"


def write_efficiency_table(table: pl.DataFrame, path: str | Path) -> None:
    """Write an efficiency table to Parquet, sorted by its key columns.

    The file is written atomically, so an interrupted write never leaves a
    partial table that a later run would reuse.
    """
    missing = [name for name in EFFICIENCY_TABLE_SCHEMA if name not in table.columns]
    if missing:
        msg = f"Efficiency table lacks the columns {missing}."
        raise ValueError(msg)
    ordered = (
        table.select(list(EFFICIENCY_TABLE_SCHEMA))
        .cast(EFFICIENCY_TABLE_SCHEMA)
        .sort(EFFICIENCY_TABLE_KEYS)
    )
    write_parquet_atomic(ordered, Path(path))


def load_efficiency_table(
    path: str | Path,
    energy_range_keV: tuple[float, float] | None = None,
) -> pl.DataFrame:
    """Read an efficiency table written by ``write_efficiency_table``.

    Parameters
    ----------
    path
        Parquet file written by the efficiencies stage (see
        ``efficiency_table_path``).
    energy_range_keV
        Optional inclusive ``(low, high)`` range of simulated energies. Only
        those rows are read, which keeps memory low for a fine mass grid.
    """
    frame = pl.scan_parquet(path)
    if energy_range_keV is not None:
        low, high = energy_range_keV
        frame = frame.filter(pl.col("sim_e").is_between(low, high))
    return frame.collect()


def primary_counts_from_table(
    table: pl.DataFrame,
) -> dict[int, dict[str, dict[str, dict[str, int]]]]:
    """Return the primary counts stored in an efficiency table.

    The efficiency table is the only place the per-run primary counts are
    kept. The result has the layout ``compute_efficiency_from_lazyframe``
    takes as ``vertex_counts``:
    ``{energy: {detector: {period: {run: n_primaries}}}}``. It covers the
    detector/runs that have rows in the table, i.e. those an efficiency was
    computed for.
    """
    rows = (
        table.select("sim_e", "detector", "period", "run", "n_primaries")
        .drop_nulls()
        .unique()
        .sort("sim_e", "detector", "period", "run")
    )
    counts: dict[int, dict[str, dict[str, dict[str, int]]]] = {}
    for energy, detector, period, run, n_primaries in rows.iter_rows():
        by_detector = counts.setdefault(int(energy), {})
        by_detector.setdefault(detector, {}).setdefault(period, {})[run] = int(
            n_primaries
        )
    return counts


def load_primary_counts(
    path: str | Path,
) -> dict[int, dict[str, dict[str, dict[str, int]]]]:
    """Read only the primary counts from an efficiency table file.

    See ``primary_counts_from_table`` for the layout of the result.
    """
    columns = ["sim_e", "detector", "period", "run", "n_primaries"]
    return primary_counts_from_table(pl.read_parquet(path, columns=columns))


def _summarize_run_entries(
    entries: Sequence[tuple[float, float, float, float]],
) -> dict[str, float] | None:
    """Summarize ``(exposure, events, primaries, FWHM syst)`` entries.

    Every entry is one detector in one run, with its own simulated events and
    primaries. The counts are combined by ``combine_run_efficiencies``, so the
    prior enters once for the whole group; ``value`` and ``unc`` are the
    posterior mean and standard deviation.
    """
    if not entries:
        return None
    values = np.asarray(entries, dtype=float)
    usable = (
        np.all(np.isfinite(values), axis=1) & (values[:, 0] > 0) & (values[:, 2] > 0)
    )
    values = values[usable]
    if len(values) == 0:
        return None
    combined = combine_run_efficiencies(values[:, 0], values[:, 1], values[:, 2])
    if combined is None:
        return None
    weights = values[:, 0]
    return {
        "value": combined["efficiency"],
        "unc": combined["efficiency_stat_unc"],
        "fwhm_syst": float(np.sum(weights / np.sum(weights) * values[:, 3])),
        "exposure": float(np.sum(weights)),
    }


def _group_membership(
    table: pl.DataFrame,
    group_by: Literal["detector_type", "detector_group"],
    detector_groups: Mapping[str, Mapping | Sequence[str]] | None,
) -> pl.DataFrame:
    """Return the groups every ``(detector, period, run)`` of a table belongs to.

    Columns: ``detector``, ``period``, ``run``, ``group``, and ``group_order``
    (the position of the group in its definition). An entry can belong to
    several detector groups.
    """
    keys = table.select("detector", "period", "run").unique(maintain_order=True)
    rows: list[tuple[str, str, str, str, int]] = []
    if group_by == "detector_type":
        order = {name: index for index, name in enumerate(DET_TYPE_MAP.values())}
        for detector, period, run in keys.iter_rows():
            detector_type = DET_TYPE_MAP.get(detector[0].upper())
            if detector_type is not None:
                rows.append(
                    (detector, period, run, detector_type, order[detector_type])
                )
    else:
        if detector_groups is None:
            msg = "detector_groups is required when group_by='detector_group'."
            raise ValueError(msg)
        group_rules: list[tuple[str, Mapping]] = []
        for group, detectors in detector_groups.items():
            if isinstance(detectors, (str, bytes)):
                msg = (
                    f"Detector group {group!r} must be a sequence or "
                    "mapping of detector names, not a string."
                )
                raise TypeError(msg)
            group_rules.append(
                (
                    group,
                    detectors
                    if isinstance(detectors, Mapping)
                    else dict.fromkeys(detectors, "all"),
                )
            )
        for detector, period, run in keys.iter_rows():
            for index, (group, rules) in enumerate(group_rules):
                if detector in rules and matches_period_run_selection(
                    period, run, rules[detector]
                ):
                    rows.append((detector, period, run, group, index))
    return pl.DataFrame(
        rows,
        schema={
            "detector": pl.String,
            "period": pl.String,
            "run": pl.String,
            "group": pl.String,
            "group_order": pl.Int64,
        },
        orient="row",
    )


def aggregate_run_aware_efficiencies(
    efficiency: pl.DataFrame | Mapping,
    selection: str,
    *,
    group_by: Literal["detector_type", "detector_group"],
    detector_groups: Mapping[str, Mapping | Sequence[str]] | None,
) -> dict[int, dict[str, dict[str, float]]]:
    """Exposure-weight run-level efficiencies into detector types or groups.

    Parameters
    ----------
    efficiency
        Efficiency table (``efficiency_table`` or ``load_efficiency_table``),
        or the nested output of ``compute_efficiency_from_lazyframe``, which
        is converted to a table first.
    selection
        Efficiency selection to aggregate.
    group_by
        ``"detector_type"`` or ``"detector_group"``.
    detector_groups
        Detector-group definitions, required for ``"detector_group"``. A
        period/run entry enters every group whose rule selects it.

    Returns
    -------
    dict
        ``{energy: {group: {"value", "unc", "fwhm_syst", "exposure"}}}``, with
        the groups in their definition order. Only valid entries with a
        positive exposure contribute. ``value`` and ``unc`` are the posterior
        mean and standard deviation from the combined counts of the group
        (``combine_run_efficiencies``); the table must hold ``n_events`` and
        ``n_primaries`` for every run.
    """
    table = (
        efficiency
        if isinstance(efficiency, pl.DataFrame)
        else efficiency_table(efficiency)
    )
    fields = [
        "exposure_kg_yr",
        "n_events",
        "n_primaries",
        "efficiency_syst_fwhm",
    ]
    selected = (
        table.filter((pl.col("selection") == selection) & (pl.col("status") == "valid"))
        .drop_nulls(fields)
        .join(
            _group_membership(table, group_by, detector_groups),
            on=["detector", "period", "run"],
            how="inner",
        )
        .sort("sim_e", "group_order", "detector", "period", "run")
    )
    output: dict[int, dict[str, dict[str, float]]] = {
        int(energy): {} for energy in table["sim_e"].unique().sort()
    }
    for (energy, group), rows in selected.group_by(
        ["sim_e", "group"], maintain_order=True
    ):
        summary = _summarize_run_entries(rows.select(fields).rows())
        if summary is not None:
            output[int(energy)][str(group)] = summary
    return output


def build_labels_dicts(
    input_dict: Mapping | pl.DataFrame,
    *,
    eres_dict: Mapping,
    group_by: Literal["detector_type", "detector_group"] = "detector_type",
    detector_groups: Mapping[str, Mapping | Sequence[str]] | None = None,
) -> dict:
    """Build a dictionary of styled plotting tuples for each valid efficiency selection.

    Parameters
    ----------
    input_dict
        Efficiency table (``load_efficiency_table``), or the nested output
        of ``compute_efficiency_from_lazyframe``. Old-format nested results
        without run-level ``period_runs`` entries are still accepted.
    eres_dict
        Nested exposure dictionary with structure
        ``{period: {run: {detector: {usability, expo, ...}}}}``.
        Used as weights only for old-format results; run-level results
        carry their own exposures.
    group_by
        Aggregate the detector results by detector type or by the supplied
        detector groups.
    detector_groups
        Detector-group definitions. Required when ``group_by="detector_group"``.
        The mapping can be loaded directly from ``groups_dict.yaml``.

    Returns
    -------
    dict
        A dictionary with the structure:
        ``{selection: (label, ls, marker, means_dict)}``
    """
    if group_by not in ("detector_type", "detector_group"):
        msg = (
            f"Unknown group_by {group_by!r}; expected 'detector_type' "
            "or 'detector_group'."
        )
        raise ValueError(msg)
    if group_by == "detector_group" and detector_groups is None:
        msg = "detector_groups is required when group_by='detector_group'."
        raise ValueError(msg)

    labels_dicts = {
        "all": ("All", "-.", "o"),
        "valid-psd": ("All - valid PSD", "-", "v"),
        "sse": ("SSE - valid PSD", "--", "s"),
        "mse": ("non-SSE - valid PSD", ":", "^"),
    }

    table: pl.DataFrame | None = None
    if isinstance(input_dict, pl.DataFrame):
        table = input_dict
    elif any(
        "period_runs" in detector_result
        for detector_results in input_dict.values()
        for detector_result in detector_results.values()
    ):
        table = efficiency_table(input_dict)

    for selection, (label, ls, marker) in list(labels_dicts.items()):
        if table is not None:
            averaged_means = aggregate_run_aware_efficiencies(
                table,
                selection,
                group_by=group_by,
                detector_groups=detector_groups,
            )
            labels_dicts[selection] = (label, ls, marker, averaged_means)
            continue

        tmp = filter_valid_selection_efficiency(input_dict, selection)
        tmp_r = restructure_efficiency_by_selection(tmp)

        if group_by == "detector_type":
            averaged_means = get_mean_fcc_det_type(
                tmp_r.get(selection, {}),
                eres_dict,
                key="efficiency",
                weight_key="expo",
                unc_key="efficiency_stat_unc",
            )
        else:
            averaged_means = get_mean_fcc_det_group(
                tmp_r.get(selection, {}),
                detector_groups,
                eres_dict,
                key="efficiency",
                weight_key="expo",
                unc_key="efficiency_stat_unc",
            )

        labels_dicts[selection] = (label, ls, marker, averaged_means)

    return labels_dicts

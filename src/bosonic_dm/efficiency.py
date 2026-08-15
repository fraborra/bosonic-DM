# Copyright (C) 2025 Francesco Borra
#

"""Detection efficiency computation for bosonic-DM analysis."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from bosonic_dm.cuts import matches_period_run_selection
from bosonic_dm.io import get_mean_fcc_det_group, get_mean_fcc_det_type
from bosonic_dm.plotting.utils import _DET_TYPE_MAP
from bosonic_dm.resolution import compute_fwhm, propagate_resolution_uncertainty
from bosonic_dm.stats import bayesian_efficiency

logger = logging.getLogger(__name__)

_SELECTION_PREFIXES = {
    "all": "all",
    "valid-psd": "valid_psd",
    "sse": "sse",
    "mse": "mse",
}


def _selection_result(
    *,
    n_events: int,
    n_events_up: int,
    n_events_down: int,
    n_primaries: int,
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
        }
    )
    return result


def _aggregate_run_selection(
    run_results: Sequence[Mapping[str, object]],
    selection: str,
) -> dict[str, object]:
    """Exposure-weight one selection after retaining its run-level results."""
    valid: list[tuple[float, Mapping[str, object]]] = []
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
            valid.append((exposure, selection_result))

    if not valid:
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
        }

    weights = np.asarray([item[0] for item in valid], dtype=float)
    normalized_weights = weights / np.sum(weights)

    def weighted(field: str) -> float:
        values = np.asarray([float(item[1][field]) for item in valid], dtype=float)
        return float(np.sum(normalized_weights * values))

    stat_uncertainties = np.asarray(
        [float(item[1]["efficiency_stat_unc"]) for item in valid], dtype=float
    )
    return {
        "status": "valid",
        "n_events": weighted("n_events"),
        "efficiency_mle": weighted("efficiency_mle"),
        "efficiency": weighted("efficiency"),
        "efficiency_stat_unc": float(
            np.sqrt(np.sum((normalized_weights * stat_uncertainties) ** 2))
        ),
        "efficiency_syst_fwhm": weighted("efficiency_syst_fwhm"),
    }


def compute_efficiency_from_lazyframe(
    lf: pl.LazyFrame,
    eres_dict: dict,
    simulated_energies: Sequence[int],
    chmap: object,
    vertex_counts: Mapping[int, Mapping[str, int]],
    half_width_fwhm: float = 2.0,
    selections: Sequence[str] = ("all", "valid-psd", "sse", "mse"),
) -> dict:
    """Compute run-aware per-detector efficiencies for all selections.

    The simulation events are collected once per energy. Each usable
    period/run/detector entry in *eres_dict* then receives its own FEP window
    from that run's resolution parameters. Run-level efficiencies are retained
    under ``period_runs`` and only then exposure-weighted into the compatibility
    detector-level ``selections`` fields.

    It collects the lazy frame *lf* once per energy and counts events inside
    each run-specific FEP window across four selections:
    - all: events in FEP window
    - valid-psd: events in FEP window with has_aoe == True
    - sse: valid-psd with is_single_site == True
    - mse: valid-psd with is_single_site == False

    Parameters
    ----------
    lf
        A Polars *lazy* scan of the parquet dataset partitioned by ``sim_e``.
        Expected columns: ``rawid``, ``energy``, ``sim_e``, ``is_good_channel``,
        ``has_aoe``, ``is_single_site``.
    eres_dict
        Nested dictionary with structure
        ``{period: {run: {detector_name: {"usability", "expo", "a", "b",
        "a_unc", "b_unc", "ab_corr"}}}}``.
    simulated_energies
        List of simulated energies (in keV) to iterate over.
    chmap
        LEGEND channel-map object (``LegendMetadata.channelmap(...)``).
    vertex_counts
        Pre-computed vertex counts, structured as
        ``{energy: {detector_name: n_vertices, ...}, ...}``.
    half_width_fwhm
        The multiplier for the FWHM to define the integration window. Defaults to 2.0.
    selections
        Selection names to compute. Supported values are ``all``,
        ``valid-psd``, ``sse``, and ``mse``.

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
                            period: {run: {"expo", "fwhm", "selections"}}
                        },
                        "selections": {
                            "all": {
                                "n_events": int,
                                "efficiency_mle": float,
                                "efficiency": float,
                                "efficiency_stat_unc": float,
                                "efficiency_syst_fwhm": float,
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
    unknown_selections = set(selections) - set(_SELECTION_PREFIXES)
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

        # The event sample is shared by all production runs; only the FEP
        # window changes with period/run calibration conditions.
        df_ene = (
            lf.filter(
                (pl.col("is_good_channel"))
                & (pl.col("sim_e") == ene)
                & (pl.col("rawid").is_in(list(rawids.values())))
            )
            .select("rawid", "energy", "has_aoe", "is_single_site")
            .collect()
        )

        # Fill nulls in booleans with False to allow safe aggregation
        df_ene = df_ene.with_columns(
            pl.col("has_aoe").fill_null(False),
            pl.col("is_single_site").fill_null(False),
        )

        for det_name, rawid in rawids.items():
            detector_frame = df_ene.filter(pl.col("rawid") == rawid)
            energies = detector_frame["energy"].to_numpy()
            distances_from_peak = np.abs(energies - ene)
            has_aoe = detector_frame["has_aoe"].to_numpy().astype(bool)
            is_single_site = detector_frame["is_single_site"].to_numpy().astype(bool)
            psd_available = bool(np.any(has_aoe)) if len(detector_frame) else None
            selection_masks = {
                "all": np.ones(len(detector_frame), dtype=bool),
                "valid-psd": has_aoe,
                "sse": has_aoe & is_single_site,
                "mse": has_aoe & ~is_single_site,
            }
            n_primaries = int(vtx_counts_for_ene.get(det_name, 0))
            period_runs: dict[str, dict[str, dict[str, object]]] = {}
            flat_run_results: list[dict[str, object]] = []

            for resolution_entry in resolution_entries[det_name]:
                period = str(resolution_entry["period"])
                run = str(resolution_entry["run"])
                fwhm = float(resolution_entry["fwhm"])
                fwhm_unc = max(float(resolution_entry["fwhm_unc"]), 0.0)
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
                                np.count_nonzero(in_window & selection_masks[selection])
                            )
                        )
                    run_selections[selection] = _selection_result(
                        n_events=counts[0],
                        n_events_up=counts[1],
                        n_events_down=counts[2],
                        n_primaries=n_primaries,
                        psd_available=psd_available,
                        selection=selection,
                    )

                run_result: dict[str, object] = {
                    "status": "valid" if n_primaries > 0 else "missing-primaries",
                    "expo": float(resolution_entry["expo"]),
                    "fwhm": fwhm,
                    "fwhm_unc": fwhm_unc,
                    "selections": run_selections,
                }
                period_runs.setdefault(period, {})[run] = run_result
                flat_run_results.append(run_result)

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


def _summarize_run_entries(
    entries: Sequence[tuple[float, float, float, float]],
) -> dict[str, float] | None:
    """Summarize ``(exposure, efficiency, stat unc, FWHM syst)`` entries."""
    if not entries:
        return None
    values = np.asarray(entries, dtype=float)
    finite = np.all(np.isfinite(values), axis=1) & (values[:, 0] > 0)
    values = values[finite]
    if len(values) == 0:
        return None
    weights = values[:, 0]
    normalized = weights / np.sum(weights)
    return {
        "value": float(np.sum(normalized * values[:, 1])),
        "unc": float(np.sqrt(np.sum((normalized * values[:, 2]) ** 2))),
        "fwhm_syst": float(np.sum(normalized * values[:, 3])),
        "exposure": float(np.sum(weights)),
    }


def _aggregate_run_aware_efficiencies(
    input_dict: Mapping,
    selection: str,
    *,
    group_by: Literal["detector_type", "detector_group"],
    detector_groups: Mapping[str, Mapping | Sequence[str]] | None,
) -> dict:
    """Aggregate only selected run-level efficiencies and their exposures."""
    output: dict = {}
    for energy, detector_results in input_dict.items():
        grouped_entries: dict[str, list[tuple[float, float, float, float]]] = {}
        for detector, detector_result in detector_results.items():
            period_runs = detector_result.get("period_runs", {})
            for period, runs in period_runs.items():
                for run, run_result in runs.items():
                    selection_result = run_result.get("selections", {}).get(
                        selection, {}
                    )
                    if selection_result.get("status") != "valid":
                        continue
                    efficiency = selection_result.get("efficiency")
                    stat_unc = selection_result.get("efficiency_stat_unc")
                    fwhm_syst = selection_result.get("efficiency_syst_fwhm")
                    exposure = run_result.get("expo")
                    if None in (efficiency, stat_unc, fwhm_syst, exposure):
                        continue
                    entry = (
                        float(exposure),
                        float(efficiency),
                        float(stat_unc),
                        float(fwhm_syst),
                    )

                    if group_by == "detector_type":
                        detector_type = _DET_TYPE_MAP.get(detector[0].upper())
                        if detector_type is not None:
                            grouped_entries.setdefault(detector_type, []).append(entry)
                        continue

                    assert detector_groups is not None
                    for group, detectors in detector_groups.items():
                        if isinstance(detectors, (str, bytes)):
                            msg = (
                                f"Detector group {group!r} must be a sequence or "
                                "mapping of detector names, not a string."
                            )
                            raise TypeError(msg)
                        group_dict = (
                            detectors
                            if isinstance(detectors, Mapping)
                            else dict.fromkeys(detectors, "all")
                        )
                        if detector not in group_dict:
                            continue
                        if matches_period_run_selection(
                            str(period), str(run), group_dict[detector]
                        ):
                            grouped_entries.setdefault(group, []).append(entry)

        output[energy] = {}
        for group, entries in grouped_entries.items():
            summary = _summarize_run_entries(entries)
            if summary is not None:
                output[energy][group] = summary
    return output


def build_labels_dicts(
    input_dict: Mapping,
    *,
    eres_dict: Mapping,
    group_by: Literal["detector_type", "detector_group"] = "detector_type",
    detector_groups: Mapping[str, Mapping | Sequence[str]] | None = None,
) -> dict:
    """Build a dictionary of styled plotting tuples for each valid efficiency selection.

    Parameters
    ----------
    input_dict
        Detector-level efficiency results.
    eres_dict
        Nested exposure dictionary with structure
        ``{period: {run: {detector: {usability, expo, ...}}}}``.
        Used as weights for the weighted average in both grouping modes.
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

    has_run_results = any(
        "period_runs" in detector_result
        for detector_results in input_dict.values()
        for detector_result in detector_results.values()
    )

    for selection, (label, ls, marker) in list(labels_dicts.items()):
        if has_run_results:
            averaged_means = _aggregate_run_aware_efficiencies(
                input_dict,
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

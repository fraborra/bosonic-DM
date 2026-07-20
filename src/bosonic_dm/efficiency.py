# Copyright (C) 2025 Francesco Borra
#

"""Detection efficiency computation for bosonic-DM analysis."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Literal

import polars as pl
from tqdm.auto import tqdm

from bosonic_dm.io import get_mean_fcc_det_group, get_mean_fcc_det_type
from bosonic_dm.stats import bayesian_efficiency

logger = logging.getLogger(__name__)

_SELECTION_PREFIXES = {
    "all": "all",
    "valid-psd": "valid_psd",
    "sse": "sse",
    "mse": "mse",
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
    """Compute per-detector efficiencies for all selections in one pass.

    For every combination of simulated energy and detector listed in
    *eres_dict*, the function uses the pre-computed *vertex_counts* as
    the number of generated primaries.

    It collects the lazy frame *lf* once per energy and uses a vectorised
    join + group-by to count events inside the FEP window for every detector
    simultaneously across four selections:
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
        ``{energy: {detector_name: {"fwhm": float, ...}, ...}, ...}``
        as produced by the resolution-extraction pipeline.
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
                        "expo":             float,   # from eres_dict
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

        if ene_key not in eres_dict:
            logger.warning("Energy %d keV not found in eres_dict, skipping", ene_key)
            continue

        det_eres = eres_dict[ene_key]
        vtx_counts_for_ene = vertex_counts.get(ene_key, {})

        # --- Phase 1: build per-detector info and read n_primaries ----------
        det_rows: list[dict] = []
        n_prim_map: dict[str, int] = {}
        expo_map: dict[str, float] = {}

        for det_name, eres_info in det_eres.items():
            try:
                rawid = chmap[det_name].daq.rawid
            except (KeyError, AttributeError):
                logger.warning("Cannot resolve rawid for %s, skipping", det_name)
                continue

            n_primaries = vtx_counts_for_ene.get(det_name, 0)
            n_prim_map[det_name] = n_primaries
            expo_map[det_name] = float(eres_info.get("expo", 0.0))

            fwhm = float(eres_info["fwhm"])
            fwhm_unc = float(eres_info.get("unc", 0.0))
            fwhm_up = fwhm + fwhm_unc
            fwhm_down = max(fwhm - fwhm_unc, 0.0)

            det_rows.append(
                {
                    "rawid": rawid,
                    "det_name": det_name,
                    "low": ene - half_width_fwhm * fwhm,
                    "high": ene + half_width_fwhm * fwhm,
                    "low_up": ene - half_width_fwhm * fwhm_up,
                    "high_up": ene + half_width_fwhm * fwhm_up,
                    "low_down": ene - half_width_fwhm * fwhm_down,
                    "high_down": ene + half_width_fwhm * fwhm_down,
                }
            )

        if not det_rows:
            continue

        det_df = pl.DataFrame(det_rows)
        known_rawids = det_df["rawid"].to_list()

        # --- Phase 2: single collect per energy -----------------------------
        df_ene = (
            lf.filter(
                (pl.col("is_good_channel"))
                & (pl.col("sim_e") == ene)
                & (pl.col("rawid").is_in(known_rawids))
            )
            .select("rawid", "energy", "has_aoe", "is_single_site")
            .collect()
        )

        # Fill nulls in booleans with False to allow safe aggregation
        df_ene = df_ene.with_columns(
            pl.col("has_aoe").fill_null(False),
            pl.col("is_single_site").fill_null(False),
        )

        # --- Phase 3: vectorised FEP counting via join + group_by -----------
        df_joined = df_ene.join(det_df, on="rawid")

        # Define selection boolean expressions
        sel_all = pl.lit(True)
        sel_psd = pl.col("has_aoe")
        sel_sse = pl.col("has_aoe") & pl.col("is_single_site")
        sel_mse = pl.col("has_aoe") & ~pl.col("is_single_site")

        # Create aggregations for each selection and variation
        aggs = []
        selection_conditions = {
            "all": sel_all,
            "valid-psd": sel_psd,
            "sse": sel_sse,
            "mse": sel_mse,
        }
        for sel_name in selections:
            prefix = _SELECTION_PREFIXES[sel_name]
            condition = selection_conditions[sel_name]
            aggs.extend(
                [
                    (
                        pl.col("energy").is_between(pl.col("low"), pl.col("high"))
                        & condition
                    )
                    .sum()
                    .alias(f"{prefix}_events"),
                    (
                        pl.col("energy").is_between(pl.col("low_up"), pl.col("high_up"))
                        & condition
                    )
                    .sum()
                    .alias(f"{prefix}_events_up"),
                    (
                        pl.col("energy").is_between(
                            pl.col("low_down"), pl.col("high_down")
                        )
                        & condition
                    )
                    .sum()
                    .alias(f"{prefix}_events_down"),
                ]
            )

        counts = df_joined.group_by("det_name").agg(
            pl.col("has_aoe").sum().alias("psd_available_events"),
            *aggs,
        )
        counts_map = {row["det_name"]: row for row in counts.iter_rows(named=True)}

        # --- Phase 4: compute efficiencies ----------------------------------
        for det_name, n_primaries in n_prim_map.items():
            row = counts_map.get(det_name)

            det_out = {
                "status": "valid" if n_primaries > 0 else "missing-primaries",
                "n_primaries": n_primaries,
                "expo": expo_map[det_name],
                "psd_available": (
                    None if row is None else bool(row["psd_available_events"] > 0)
                ),
                "selections": {},
            }

            for sel_name in selections:
                prefix = _SELECTION_PREFIXES[sel_name]
                n_events = row[f"{prefix}_events"] if row else 0
                n_events_up = row[f"{prefix}_events_up"] if row else 0
                n_events_down = row[f"{prefix}_events_down"] if row else 0

                selection_out = {
                    "status": "valid",
                    "n_events": n_events,
                    "efficiency_mle": None,
                    "efficiency": None,
                    "efficiency_stat_unc": None,
                    "efficiency_syst_fwhm": None,
                }

                if n_primaries <= 0:
                    selection_out["status"] = "missing-primaries"
                elif sel_name != "all" and det_out["psd_available"] is False:
                    selection_out["status"] = "psd-unavailable"
                elif any(
                    count < 0 or count > n_primaries
                    for count in (n_events, n_events_up, n_events_down)
                ):
                    selection_out["status"] = "invalid-counts"
                    logger.warning(
                        "Invalid counts for %s at %d keV (%s): "
                        "nominal=%d, up=%d, down=%d, primaries=%d",
                        det_name,
                        ene_key,
                        sel_name,
                        n_events,
                        n_events_up,
                        n_events_down,
                        n_primaries,
                    )
                else:
                    ratio, ratio_sigma = bayesian_efficiency(n_events, n_primaries)
                    ratio_up, _ = bayesian_efficiency(n_events_up, n_primaries)
                    ratio_down, _ = bayesian_efficiency(n_events_down, n_primaries)
                    selection_out.update(
                        {
                            "efficiency_mle": float(n_events / n_primaries),
                            "efficiency": ratio,
                            "efficiency_stat_unc": ratio_sigma,
                            "efficiency_syst_fwhm": abs(ratio_up - ratio_down) / 2.0,
                        }
                    )

                det_out["selections"][sel_name] = selection_out

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


def build_labels_dicts(
    input_dict: Mapping,
    *,
    group_by: Literal["detector_type", "detector_group"] = "detector_type",
    detector_groups: Mapping[str, Mapping | Sequence[str]] | None = None,
    eres_dict: Mapping | None = None,
) -> dict:
    """Build a dictionary of styled plotting tuples for each valid efficiency selection.

    Parameters
    ----------
    input_dict
        Detector-level efficiency results.
    group_by
        Aggregate the detector results by detector type or by the supplied
        detector groups.
    detector_groups
        Detector-group definitions. Required when ``group_by="detector_group"``.
        The mapping can be loaded directly from ``groups_dict.yaml``.
    eres_dict
        Period/run detector exposure dictionary. Required when
        ``group_by="detector_group"``.

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
    if group_by == "detector_group" and (
        detector_groups is None or eres_dict is None
    ):
        msg = (
            "detector_groups and eres_dict are required when "
            "group_by='detector_group'."
        )
        raise ValueError(msg)

    labels_dicts = {
        "all": ("All", "-.", "o"),
        "valid-psd": ("All - valid PSD", "-", "v"),
        "sse": ("SSE - valid PSD", "--", "s"),
        "mse": ("MSE - valid PSD", ":", "^"),
    }

    for selection, (label, ls, marker) in list(labels_dicts.items()):
        tmp = filter_valid_selection_efficiency(input_dict, selection)
        tmp_r = restructure_efficiency_by_selection(tmp)

        if group_by == "detector_type":
            averaged_means = get_mean_fcc_det_type(
                tmp_r.get(selection, {}),
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

# Copyright (C) 2025 Francesco Borra
#

"""Data I/O utilities for bosonic-DM analysis."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import awkward as ak
import lh5
import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm

from bosonic_dm.cuts import compute_group_exposure
from bosonic_dm.plotting.utils import _DET_TYPE_MAP
from bosonic_dm.stats import compute_weighted_uncertainty, weighted_mean

logger = logging.getLogger(__name__)


def ak_to_pandas(
    ak_obj1: ak.Array,
    ak_obj2: ak.Array,
    library: str = "pandas",
) -> pd.DataFrame | pl.DataFrame:
    """Convert Awkward Array objects into a flat Pandas or Polars DataFrame.

    This function extracts data from Awkward objects (data.geds and data.trigger)
    for single-multiplicity events (multiplicity == 1) and converts them into a
    flat DataFrame of the requested library.

    Parameters
    ----------
    ak_obj1 : ak.Array
        Awkward object containing Germanium detector data (geds), filtered for
        multiplicity == 1 events. It contains event-level variables (e.g. energy_sum,
        multiplicity) and channel/hit-level nested arrays (e.g. energy, hit_idx).
    ak_obj2 : ak.Array
        Awkward object containing trigger data, filtered for multiplicity == 1 events.
        It contains global event/trigger variables (e.g. evtid, period, run).
    library : {"pandas", "polars"}, default="pandas"
        The DataFrame library to return.

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        DataFrame with one row per event and columns for each variable.
    """
    data = {
        # For nested fields in geds (ak_obj1), use ak.flatten to remove the nested
        # list structure (since multiplicity == 1, there is exactly one value per
        # event/list) and ak.to_numpy for the final conversion.
        "energy": ak.to_numpy(ak.flatten(ak_obj1.energy)),
        "energy_sum": ak.to_numpy(
            ak_obj1.energy_sum
        ),  # Already flat at event level, no flatten needed
        "hit_idx": ak.to_numpy(ak.flatten(ak_obj1.hit_idx)),
        "aoe": ak.to_numpy(ak.flatten(ak_obj1.aoe)),
        "has_aoe": ak.to_numpy(ak.flatten(ak_obj1.has_aoe)),
        "is_good_channel": ak.to_numpy(ak.flatten(ak_obj1.is_good_channel)),
        "is_single_site": ak.to_numpy(ak.flatten(ak_obj1.is_single_site)),
        "multiplicity": ak.to_numpy(
            ak_obj1.multiplicity
        ),  # Already flat at event level, no flatten needed
        "rawid": ak.to_numpy(ak.flatten(ak_obj1.rawid)),
        # For trigger fields (ak_obj2), variables are already flat at event level
        # and do not have nested structures, so use ak.to_numpy directly.
        "evtid": ak.to_numpy(ak_obj2.evtid),
        "period": ak.to_numpy(ak_obj2.period),
        "run": ak.to_numpy(ak_obj2.run),
    }

    if library.lower() == "pandas":
        return pd.DataFrame(data)
    if library.lower() == "polars":
        return pl.DataFrame(data)

    msg = f"Unknown library '{library}'; expected 'pandas' or 'polars'"
    raise ValueError(msg)


def get_rawid_lists(chmap, rawids):
    rawid_by_det_type = {"ICPC": [], "BEGe": [], "PPC": [], "COAX": []}

    for rid in np.unique(rawids):
        ge = chmap.map("daq.rawid")[rid]["name"]
        rawid_by_det_type[_DET_TYPE_MAP[ge[0]]].append(rid)

    return rawid_by_det_type


def get_rawids_map(chmap, ges):
    rawids_map = {}

    for ge in ges:
        rawids_map[ge] = chmap[ge].daq.rawid

    return rawids_map


def get_values_sorted(
    det_dict: Mapping[str, dict],
    ges_sorted: Sequence[str],
    key: str = "ratio",
) -> tuple[list[str], list[float]]:
    """Extract values from a dictionary according to a specific detector order.

    Parameters
    ----------
    det_dict
        Dictionary containing per-detector data.
        Example: ``{'V02160A': {'ratio': 0.9, ...}, ...}``
    ges_sorted
        List of detector names specifying the desired order.
    key
        Inner key whose value is extracted for each detector.

    Returns
    -------
    tuple[list[str], list[float]]
        A tuple containing the list of detectors and the corresponding
        values ordered as requested. If a detector is not found
        in *det_dict* or the key is missing, the value defaults to ``nan``.
    """
    values = []
    keys = []

    for ge in ges_sorted:
        keys.append(ge)
        if ge in det_dict:
            values.append(det_dict[ge].get(key, np.nan))
        else:
            values.append(np.nan)

    return keys, values


def get_mean_fcc_det_type(
    ratio_dict: Mapping,
    key: str = "ratio",
    weight_key: str = "expo",
    unc_key: str | None = None,
    exclude_dets: Sequence[str] | None = None,
) -> dict:
    """Compute per-detector-type weighted mean of a nested dict.

    Parameters
    ----------
    ratio_dict
        Nested dictionary with structure
        ``{ene: {ge: {key: value, unc_key: uncertainty, ...}, ...}, ...}``.
    key
        Inner key whose value is averaged across detectors of the same type.
    weight_key
        Inner key used as weight for the weighted average.
    unc_key
        Inner key used as the uncertainty for each detector's value. If provided,
        the uncertainty is propagated using `compute_weighted_uncertainty` and
        the function returns a dict ``{"value": mean, "unc": unc_total}`` instead
        of just a float.
    exclude_dets
        List of detector names to exclude from the average.
    """
    ratio_dict_means: dict = {}

    for ene, ge_dict in ratio_dict.items():
        ratio_dict_means[ene] = {}

        type_data: dict[str, dict[str, list]] = {
            label: {"vals": [], "w": [], "unc": []} for label in _DET_TYPE_MAP.values()
        }

        for ge, data_dict in ge_dict.items():
            if exclude_dets is not None and ge in exclude_dets:
                continue

            prefix = ge[0].upper()
            det_type = _DET_TYPE_MAP.get(prefix)
            if det_type is None:
                continue

            val = data_dict.get(key)
            w = data_dict.get(weight_key)
            if val is None or w is None:
                continue

            type_data[det_type]["vals"].append(val)
            type_data[det_type]["w"].append(w)
            if unc_key is not None:
                type_data[det_type]["unc"].append(data_dict.get(unc_key, 0.0))

        for det_type, d in type_data.items():
            if not d["vals"]:
                continue

            vals_arr = np.array(d["vals"], dtype=float)
            w_arr = np.array(d["w"], dtype=float)

            # Mask to drop non-finite values/weights and keep arrays aligned
            mask = np.isfinite(vals_arr) & np.isfinite(w_arr)
            vals_arr = vals_arr[mask]
            w_arr = w_arr[mask]

            if len(vals_arr) == 0 or np.sum(w_arr) == 0:
                continue

            mean = weighted_mean(vals_arr, w_arr)
            exposure_total = float(np.sum(w_arr))

            if unc_key is not None:
                s_arr = np.array(d["unc"], dtype=float)[mask]
                s_arr = np.where(np.isfinite(s_arr), s_arr, 0.0)

                unc_total = compute_weighted_uncertainty(w_arr, vals_arr, mean, s_arr)
                ratio_dict_means[ene][det_type] = {
                    "value": mean,
                    "unc": unc_total,
                    "exposure": exposure_total,
                }
            else:
                ratio_dict_means[ene][det_type] = mean

    return ratio_dict_means


def get_mean_fcc_det_group(
    ratio_dict: Mapping,
    detector_groups: Mapping[str, Mapping | Sequence[str]],
    eres_dict: Mapping,
    key: str = "ratio",
    weight_key: str = "expo",
    unc_key: str | None = None,
    exclude_dets: Sequence[str] | None = None,
) -> dict:
    """Compute a weighted mean for each supplied detector group.

    Parameters
    ----------
    ratio_dict
        Nested dictionary with structure
        ``{ene: {ge: {key: value, weight_key: weight, ...}, ...}, ...}``.
    detector_groups
        Mapping from each output group name to either a sequence of detector
        names or a mapping whose keys are detector names. The latter accepts
        the structure loaded directly from ``groups_dict.yaml`` and applies
        its period/run inclusion and exclusion rules. A sequence of detector
        names selects all periods and runs for those detectors.
    eres_dict
        Nested exposure dictionary with structure
        ``{period: {run: {detector: {"usability": ..., weight_key: ...}}}}``.
        Only usable period/run entries selected for a detector's group are
        included in that detector's weight.
    key
        Inner key whose value is averaged across detectors in the same group.
    weight_key
        Exposure key in *eres_dict* used as weight for the weighted average.
    unc_key
        Inner key used as the uncertainty for each detector's value. If provided,
        the uncertainty is propagated using `compute_weighted_uncertainty` and
        the function returns a dict ``{"value": mean, "unc": unc_total,
        "exposure": exposure_total}`` instead of just a float.
    exclude_dets
        List of detector names to exclude from every group average.

    Notes
    -----
    Groups are evaluated independently. A detector included in more than one
    group contributes to each average with the exposure selected for that
    particular group.
    """
    detector_to_groups: dict[str, dict[str, float]] = {}
    for group, detectors in detector_groups.items():
        if isinstance(detectors, (str, bytes)):
            msg = (
                f"Detector group {group!r} must be a sequence or mapping of "
                "detector names, not a string."
            )
            raise TypeError(msg)

        group_dict = (
            detectors
            if isinstance(detectors, Mapping)
            else {detector: "all" for detector in detectors}
        )
        for detector, selection in group_dict.items():
            exposure = compute_group_exposure(
                eres_dict,
                {detector: selection},
                exposure_key=weight_key,
            )
            detector_to_groups.setdefault(detector, {})[group] = exposure

    ratio_dict_means: dict = {}

    for ene, ge_dict in ratio_dict.items():
        ratio_dict_means[ene] = {}
        group_data: dict[str, dict[str, list]] = {
            group: {"vals": [], "w": [], "unc": []} for group in detector_groups
        }

        for ge, data_dict in ge_dict.items():
            if exclude_dets is not None and ge in exclude_dets:
                continue

            group_exposures = detector_to_groups.get(ge)
            if group_exposures is None:
                continue

            val = data_dict.get(key)
            if val is None:
                continue

            for group, exposure in group_exposures.items():
                group_data[group]["vals"].append(val)
                group_data[group]["w"].append(exposure)
                if unc_key is not None:
                    group_data[group]["unc"].append(data_dict.get(unc_key, 0.0))

        for group, data in group_data.items():
            if not data["vals"]:
                continue

            vals_arr = np.array(data["vals"], dtype=float)
            w_arr = np.array(data["w"], dtype=float)

            # Mask to drop non-finite values/weights and keep arrays aligned
            mask = np.isfinite(vals_arr) & np.isfinite(w_arr)
            vals_arr = vals_arr[mask]
            w_arr = w_arr[mask]

            if len(vals_arr) == 0 or np.sum(w_arr) == 0:
                continue

            mean = weighted_mean(vals_arr, w_arr)
            exposure_total = float(np.sum(w_arr))

            if unc_key is not None:
                s_arr = np.array(data["unc"], dtype=float)[mask]
                s_arr = np.where(np.isfinite(s_arr), s_arr, 0.0)
                unc_total = compute_weighted_uncertainty(
                    w_arr, vals_arr, mean, s_arr
                )
                ratio_dict_means[ene][group] = {
                    "value": mean,
                    "unc": unc_total,
                    "exposure": exposure_total,
                }
            else:
                ratio_dict_means[ene][group] = mean

    return ratio_dict_means


def build_parquet_dataset(
    *,
    energies: Sequence[int],
    cvt_files: Mapping[int, Sequence[Path]],
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    """Build a parquet dataset partitioned by simulated energy.

    For each energy, the function reads the event data from the
    provided CVT files, selects multiplicity-1 events, appends
    `sim_e` and `coincident_spms` columns, and concatenates the
    results into a partitioned Parquet dataset.

    Parameters
    ----------
    energies : Sequence[int]
        List of simulated energies in keV.
    cvt_files : Mapping[int, Sequence[Path]]
        Mapping from energy to a sequence of LH5 CVT file paths.
    output_dir : Path
        Final name of the parquet output directory.
    overwrite : bool, default=False
        If True, remove the output files if they already exist.
        If False and an output file exists, an exception is raised.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ene in tqdm(energies):
        files_for_ene = cvt_files.get(ene, [])
        if not files_for_ene:
            logger.warning("No CVT files provided for %d keV", ene)
            continue

        partition_dir = output_dir / f"sim_e={ene}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        # Using a deterministic output filename per energy
        outfile = partition_dir / "data.parquet"

        if outfile.exists() and not overwrite:
            msg = f"File already exists: {outfile}"
            raise FileExistsError(msg)

        if overwrite and outfile.exists():
            outfile.unlink()

        dfs = []
        for cvt_path in files_for_ene:
            cvt_file = str(cvt_path)
            data = lh5.read_as(
                "evt",
                cvt_file,
                field_mask=["coincident", "geds", "trigger"],
                library="ak",
            )

            mult1_mask = data.geds.multiplicity == 1
            tmp = data.geds[mult1_mask]
            tmp2 = data.trigger[mult1_mask]
            pl_df = ak_to_pandas(tmp, tmp2, library="polars")

            # Store the SiPM coincidence flag instead of cutting on it
            spms_col = ak.to_numpy(data.coincident.spms[mult1_mask])
            pl_df = pl_df.with_columns(
                pl.Series("coincident_spms", spms_col),
                pl.lit(ene).alias("sim_e"),
            )
            dfs.append(pl_df)

        if dfs:
            final_df = pl.concat(dfs)
            final_df.write_parquet(outfile)

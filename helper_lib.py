from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from lgdo import lh5
from tqdm.notebook import tqdm

logger = logging.getLogger(__name__)

_DET_TYPE_MAP = {"B": "BEGe", "C": "COAX", "V": "ICPC", "P": "PPC"}
_DET_TYPE_COLOR = {
    "BEGe": "tab:blue",
    "COAX": "tab:orange",
    "ICPC": "tab:green",
    "PPC": "tab:red",
}


def select_channel(energies, channels, rawid):
    return energies[channels == rawid]


def expand_range(item):
    if ".." not in item:
        return [item]  # already single
    start, end = item.split("..")

    prefix = start[0]  # 'r'
    s = int(start[1:])
    e = int(end[1:])
    width = len(start) - 1  # number of digits

    return [f"{prefix}{i:0{width}d}" for i in range(s, e + 1)]


def prendi_ene_rawid(cvt_file):
    data_ak = lh5.read_as(
        "/evt", cvt_file, field_mask=["coincident", "geds", "trigger"], library="ak"
    )

    data_in_ge = data_ak[data_ak.coincident.geds]

    energies = ak.flatten(data_in_ge.geds.energy[data_in_ge.geds.multiplicity == 1])
    channels = ak.flatten(data_in_ge.geds.rawid[data_in_ge.geds.multiplicity == 1])

    return energies, channels


def riempi_dict(cvt_file, ges, chmap, stp_files, ene, thr):
    energies, channels = prendi_ene_rawid(cvt_file)

    det_ene = {}

    for ge in tqdm(ges):
        rawid = chmap[ge].daq.rawid
        stp_ge = lh5.read_as(f"/stp/{ge}", stp_files, library="ak")
        det_ene[ge] = {}
        det_ene[ge]["energy"] = energies[channels == rawid]
        det_ene[ge]["evtids"] = stp_ge.evtid
        tmp = det_ene[ge]["energy"]
        det_ene[ge]["ratio"] = len(tmp[(tmp < ene + thr) & (tmp > ene - thr)]) / len(
            det_ene[ge]["evtids"]
        )

    return det_ene


def get_n_primaries(ges, stp_files):
    det_prim = {}

    for ge in tqdm(ges):
        stp_ge = lh5.read_as(f"/stp/{ge}", stp_files, library="ak")
        det_prim[ge] = len(stp_ge.evtid)

    return det_prim


def compute_ratio(det_ene, ene, thr):
    for ge in det_ene.keys():
        tmp = det_ene[ge]["energy"]
        det_ene[ge]["ratio"] = len(tmp[(tmp < ene + thr) & (tmp > ene - thr)]) / len(
            det_ene[ge]["evtids"]
        )


def prendi_valori(det_ene):
    keys = list(det_ene.keys())

    values = []
    for ge in keys:
        values.append(det_ene[ge]["ratio"])

    return keys, values


def plot_e_det_type(ene_dict, ene, bins=300, lw=1):
    bege = ak.Array([])
    coax = ak.Array([])
    icpc = ak.Array([])
    ppc = ak.Array([])

    for ge in ene_dict.keys():
        if ge[0] == "B":
            bege = ak.concatenate([bege, ene_dict[ge]["energy"]])

        if ge[0] == "C":
            coax = ak.concatenate([coax, ene_dict[ge]["energy"]])

        if ge[0] == "V":
            icpc = ak.concatenate([icpc, ene_dict[ge]["energy"]])

        if ge[0] == "P":
            ppc = ak.concatenate([ppc, ene_dict[ge]["energy"]])

    plt.figure(figsize=(10, 6))
    plt.hist(bege, bins=bins, label="BEGe", histtype="step", linewidth=lw)
    plt.hist(ppc, bins=bins, label="PPC", histtype="step", linewidth=lw)
    plt.hist(coax, bins=bins, label="COAX", histtype="step", linewidth=lw)
    plt.hist(icpc, bins=bins, label="ICPC", histtype="step", linewidth=lw)
    plt.yscale("log")
    plt.legend(title=f"{ene}keV e-", fontsize=13)
    plt.xlabel("Processed Energy [keV]", fontsize=13)
    plt.savefig(f"notebooks/plots/det_type_energy_{ene}.png", dpi=300)
    plt.show()


def get_rawid_lists(chmap, rawids):
    rawid_by_det_type = {"ICPC": [], "BEGe": [], "PPC": [], "COAX": []}

    for rid in np.unique(rawids):
        ge = chmap.map("daq.rawid")[rid]["name"]
        rawid_by_det_type[_DET_TYPE_MAP[ge[0]]].append(rid)

    return rawid_by_det_type


def get_values_type(det_dict, ene, bins, lw=1):
    bege = ak.Array([])
    coax = ak.Array([])
    icpc = ak.Array([])
    ppc = ak.Array([])

    for ge in det_dict.keys():
        if ge[0] == "B":
            bege = ak.concatenate([bege, det_dict[ge]["energy"]])

        if ge[0] == "C":
            coax = ak.concatenate([coax, det_dict[ge]["energy"]])

        if ge[0] == "V":
            icpc = ak.concatenate([icpc, det_dict[ge]["energy"]])

        if ge[0] == "P":
            ppc = ak.concatenate([ppc, det_dict[ge]["energy"]])

    plt.figure(figsize=(10, 6))
    plt.hist(bege, bins=bins, label="BEGe", histtype="step", linewidth=lw)
    plt.hist(ppc, bins=bins, label="PPC", histtype="step", linewidth=lw)
    plt.hist(coax, bins=bins, label="COAX", histtype="step", linewidth=lw)
    plt.hist(icpc, bins=bins, label="ICPC", histtype="step", linewidth=lw)
    plt.yscale("log")
    plt.legend(title=f"{ene}keV e-", fontsize=13)
    plt.xlabel("Processed Energy [keV]", fontsize=13)
    plt.savefig(f"notebooks/plots/det_type_energy_{ene}.png", dpi=300)
    plt.show()


def get_mean_fcc_det_type(
    ratio_dict: dict,
    key: str = "ratio",
    weight_key: str = "expo",
) -> dict:
    """Compute per-detector-type weighted mean of a nested dict.

    Parameters
    ----------
    ratio_dict
        Nested dictionary with structure
        ``{ene: {ge: {key: value, weight_key: weight, ...}, ...}, ...}``.
    key
        Inner key whose value is averaged across detectors of the same type.
    weight_key
        Inner key used as weight for the weighted average.
    """
    ratio_dict_means: dict = {}

    for ene, ge_dict in ratio_dict.items():
        ratio_dict_means[ene] = {}

        type_data: dict[str, dict[str, list]] = {
            label: {"vals": [], "w": []} for label in _DET_TYPE_MAP.values()
        }

        for ge, data_dict in ge_dict.items():
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

        for det_type, d in type_data.items():
            vals_arr = clean_array(d["vals"])
            w_arr = clean_array(d["w"])

            # keep only entries where both value and weight survived cleaning
            min_len = min(len(vals_arr), len(w_arr))
            vals_arr = vals_arr[:min_len]
            w_arr = w_arr[:min_len]

            ratio_dict_means[ene][det_type] = weighted_mean(vals_arr, w_arr)

    return ratio_dict_means


def clean_array(arr):
    arr = np.asarray(arr)
    arr = arr.astype(float)
    return arr[(arr != 0) & np.isfinite(arr)]


def weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute the weighted mean ``sum(w * v) / sum(w)``.

    Parameters
    ----------
    values
        Array of values.
    weights
        Array of weights (same length as *values*).

    Returns
    -------
    float
        Weighted mean, or ``nan`` if the total weight is zero or arrays are empty.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    total_w = np.sum(weights)
    if total_w == 0 or len(values) == 0:
        return np.nan
    return float(np.sum(weights * values) / total_w)


def compute_weighted_uncertainty(
    w_arr: np.ndarray,
    vals_arr: np.ndarray,
    mean_val: float,
    s_arr: np.ndarray,
) -> float:
    """Compute the total weighted uncertainty including measurement and scatter components.

    Parameters
    ----------
    w_arr
        Array of weights.
    vals_arr
        Array of values.
    mean_val
        Weighted mean of the values.
    s_arr
        Array of uncertainties for each value.

    Returns
    -------
    float
        Total combined uncertainty (measurement ⊕ scatter).
    """
    total_w = np.sum(w_arr)
    if total_w == 0:
        return 0.0

    # measurement component: quadrature propagation through weighted average
    unc_meas = float(np.sqrt(np.sum((w_arr * s_arr) ** 2)) / total_w)

    # scatter component: weighted variance across entries
    var_w = float(np.sum(w_arr * (vals_arr - mean_val) ** 2) / total_w)
    n_eff = float(total_w**2 / np.sum(w_arr**2))
    unc_scatter = float(np.sqrt(var_w / n_eff)) if n_eff > 0 else 0.0

    return float(np.sqrt(unc_meas**2 + unc_scatter**2))


def ak_to_pandas(ak_obj1, ak_obj2):
    df = pd.DataFrame()

    df["energy"] = ak.to_numpy(ak.flatten(ak_obj1.energy))
    df["energy_sum"] = ak.to_numpy(ak_obj1.energy_sum)
    df["hit_idx"] = ak.to_numpy(ak.flatten(ak_obj1.hit_idx))
    df["is_good_channel"] = ak.to_numpy(ak.flatten(ak_obj1.is_good_channel))
    df["is_single_site"] = ak.to_numpy(ak.flatten(ak_obj1.is_single_site))
    df["multiplicity"] = ak.to_numpy(ak_obj1.multiplicity)
    df["rawid"] = ak.to_numpy(ak.flatten(ak_obj1.rawid))

    df["evtid"] = ak.to_numpy(ak_obj2.evtid)
    df["period"] = ak.to_numpy(ak_obj2.period)
    df["run"] = ak.to_numpy(ak_obj2.run)

    return df


def get_rawids_map(chmap, ges):
    rawids_map = {}

    for ge in ges:
        rawids_map[ge] = chmap[ge].daq.rawid

    return rawids_map


def get_values_sorted(
    det_dict: Mapping[str, dict], ges_sorted: Sequence[str]
) -> tuple[list[str], list[float]]:
    """Extract ratio values from a dictionary according to a specific detector order.

    Parameters
    ----------
    det_dict
        Dictionary containing ratio values for each detector.
        Example: ``{'V02160A': {'ratio': 0.9, ...}, ...}``
    ges_sorted
        List of detector names specifying the desired order.

    Returns
    -------
    tuple[list[str], list[float]]
        A tuple containing the list of detectors and the corresponding
        ratio values ordered as requested. If a detector is not found
        in *det_dict*, its ratio defaults to ``nan``.
    """
    values = []
    keys = []

    for ge in ges_sorted:
        keys.append(ge)
        if ge in det_dict:
            values.append(det_dict[ge].get("ratio", np.nan))
        else:
            values.append(np.nan)

    return keys, values


## NEW VERSION


def build_parquet_dataset(
    simulated_energies: list[int],
    scratch_folder: str | Path,
    job_base: str,
    outdir_name: str,
    overwrite: bool = False,
) -> None:
    """Build a parquet dataset partitioned by simulated energy.

    For each energy in `simulated_energies`, the function:
      - locates the corresponding LH5 file,
      - reads the event data,
      - selects multiplicity-1 events,
      - converts awkward arrays to a pandas DataFrame,
      - appends a `sim_e` column,
      - concatenates all energies together,
      - writes the final dataset as partitioned parquet.

    Parameters
    ----------
    simulated_energies : list[int]
        List of simulated energies in keV.
        Example:
            [500, 1000, 1500]

    scratch_folder : str | Path
        Base scratch directory containing the generated LH5 files.

    job_base : str
        Job string template containing the `{tag}` placeholder.
        Example:
            "fromfile_dark_compton_{tag}_hpge_bulk"

        The placeholder is replaced with:
            tag = f"{energy}keV"

    outdir_name : str
        Final name of the parquet output directory.
        Example:
            "dark-compton"

        The dataset will be written to:
            ./v1/parquet/{outdir_name}

    overwrite : bool, default=False
        If True, remove the output directory if it already exists.
        If False and the output directory exists, an exception
        is raised.

    Returns
    -------
    None
        The function writes the parquet dataset to disk and
        does not return any object.
    """
    outdir = Path(f"./v1/parquet/{outdir_name}")
    outdir.mkdir(parents=True, exist_ok=True)

    for ene in tqdm(simulated_energies):
        tag = f"{ene}keV"

        job_string = job_base.format(tag=tag)

        # If outfile already exist and overwrite=False exit
        partition_dir = outdir / f"sim_e={ene}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        outfile = partition_dir / f"{job_string}.parquet"

        if outfile.exists() and not overwrite:
            msg = f"File already exists: {outfile}"
            raise FileExistsError(msg)

        if overwrite and outfile.exists():
            outfile.unlink()

        # Search for simulation output file
        search_dir = Path(scratch_folder) / "generated" / "tier" / "cvt"
        filename_pattern = f"l200cfg01-{job_string}-tier_cvt.lh5"
        matches = list(search_dir.glob(filename_pattern))

        if len(matches) == 0:
            logger.warning("File not found for %d keV", ene)
            continue

        cvt_file = str(matches[0])

        data = lh5.read_as(
            "evt",
            cvt_file,
            field_mask=["coincident", "geds", "trigger"],
            library="ak",
        )

        tmp = data.geds[data.geds.multiplicity == 1]
        tmp2 = data.trigger[data.geds.multiplicity == 1]

        df = ak_to_pandas(tmp, tmp2)

        df["sim_e"] = np.full(len(df), ene, dtype=int)

        pl_df = pl.from_pandas(df, include_index=False)

        pl_df.write_parquet(outfile)


def compute_ratio_from_lazyframe(
    lf: pl.LazyFrame,
    eres_dict: dict,
    simulated_energies: Sequence[int],
    rawid_by_det_type: Mapping[str, Sequence[int]],
    chmap: object,
) -> dict:
    """Compute per-detector efficiency ratios from a Polars lazy scan.

    For every combination of simulated energy and detector listed in
    *eres_dict*, the function:

    1. Looks up the detector-specific FWHM from *eres_dict*.
    2. Defines the full-energy-peak integration window as
       ``[e_value - 2·FWHM, e_value + 2·FWHM]``.
    3. Filters the lazy frame *lf* (collecting only the needed rows)
       to count the events inside the window (``n_events``) and the
       total number of good-channel events for that detector and
       simulated energy (``n_primaries``).
    4. Computes ``ratio = n_events / n_primaries``
       (or ``nan`` when ``n_primaries == 0``).

    Parameters
    ----------
    lf
        A Polars *lazy* scan of the parquet dataset.  Expected columns:
        ``rawid``, ``energy``, ``sim_e``, ``is_good_channel``.
    eres_dict
        Nested dictionary with structure
        ``{energy: {detector_name: {"fwhm": float, ...}, ...}, ...}``
        as produced by the resolution-extraction pipeline
        (e.g. ``eres_per_det_tot.yaml``).
    simulated_energies
        List of simulated energies (in keV) to iterate over.
    rawid_by_det_type
        Mapping ``{det_type: [rawid, ...], ...}`` used only to build
        the inverse map rawid → detector name when *chmap* is not
        available for a given rawid.
    chmap
        LEGEND channel-map object (``LegendMetadata.channelmap(...)``).
        Must support ``chmap.map("daq.rawid")[rawid]["name"]``.

    Returns
    -------
    dict
        Nested dictionary with structure::

            {
                energy: {
                    detector_name: {
                        "n_events":    int,
                        "n_primaries": int,
                        "ratio":       float,
                        "expo":        float,   # from eres_dict
                    },
                    ...
                },
                ...
            }
    """
    # Build an inverse map: rawid → detector name for quick lookup
    all_rawids: list[int] = []
    for rids in rawid_by_det_type.values():
        all_rawids.extend(rids)

    rawid_to_name: dict[int, str] = {}
    for rid in all_rawids:
        try:
            rawid_to_name[rid] = chmap.map("daq.rawid")[rid]["name"]
        except (KeyError, TypeError):
            continue

    ratio_dict: dict = {}

    for ene in tqdm(simulated_energies):
        ene_key = int(ene)
        ratio_dict[ene_key] = {}

        # Skip energies not present in the resolution dictionary
        if ene_key not in eres_dict:
            logger.warning("Energy %d keV not found in eres_dict, skipping", ene_key)
            continue

        det_eres = eres_dict[ene_key]

        for det_name, eres_info in det_eres.items():
            fwhm = float(eres_info["fwhm"])
            low = ene - 2.0 * fwhm
            high = ene + 2.0 * fwhm

            # Resolve rawid for this detector
            rawid: int | None = None
            try:
                rawid = chmap[det_name].daq.rawid
            except (KeyError, AttributeError):
                logger.warning("Cannot resolve rawid for %s, skipping", det_name)
                continue

            # Collect counts from the lazy frame.
            # First: all good-channel events for this detector + energy
            df_filtered = lf.filter(
                (pl.col("rawid") == rawid)
                & (pl.col("is_good_channel"))
                & (pl.col("sim_e") == ene)
            ).collect()

            n_primaries: int = len(df_filtered)

            # Second: events inside the FEP window
            n_events: int = int(
                df_filtered.filter(pl.col("energy").is_between(low, high)).height
            )

            ratio: float = n_events / n_primaries if n_primaries > 0 else np.nan

            ratio_dict[ene_key][det_name] = {
                "n_events": n_events,
                "n_primaries": n_primaries,
                "ratio": ratio,
                "expo": float(eres_info.get("expo", 0.0)),
            }

    return ratio_dict

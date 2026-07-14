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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
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
        ``{ene: {ge: {key: value, weight_key: weight, ...}, ...}, ...}``.
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


def bayesian_efficiency(
    k: int,
    n: int,
    alpha0: float = 0.5,
    beta0: float = 0.5,
) -> tuple[float, float]:
    """Compute Bayesian estimate of binomial efficiency with a Beta conjugate prior.

    Parameters
    ----------
    k
        Number of successes (events in the FEP window).
    n
        Number of trials (total good-channel events).
    alpha0
        First shape parameter of the Beta prior (default: Jeffrey's 0.5).
    beta0
        Second shape parameter of the Beta prior (default: Jeffrey's 0.5).

    Returns
    -------
    tuple[float, float]
        Posterior mean (ratio) and posterior standard deviation (ratio uncertainty).
    """
    alpha = alpha0 + k
    beta = beta0 + n - k

    mean = alpha / (alpha + beta)
    var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

    return mean, float(np.sqrt(var))


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


## NEW VERSION


def build_parquet_dataset(
    simulated_energies: list[int],
    scratch_folder: str | Path,
    job_base: str,
    outdir_name: str | Path,
    datasets_outdir: str | Path = "./v1/parquet",
    overwrite: bool = False,
) -> None:
    """Build a parquet dataset partitioned by simulated energy.

    For each energy in `simulated_energies`, the function:
      - locates the corresponding LH5 file,
      - reads the event data,
      - selects multiplicity-1 events,
      - converts awkward arrays to a pandas DataFrame,
      - appends a `sim_e` column and a `coincident_spms` column,
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

    outdir_name : str | Path
        Final name of the parquet output directory.

    datasets_outdir : str | Path, default="./v1/parquet"
        Base directory for the parquet dataset.
        The output dataset will be written to:
            {datasets_outdir}/{outdir_name}

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
    outdir = Path(f"{datasets_outdir}/{outdir_name}")
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

        pl_df.write_parquet(outfile)


def compute_efficiency_from_lazyframe(
    lf: pl.LazyFrame,
    eres_dict: dict,
    simulated_energies: Sequence[int],
    chmap: object,
    scratch_folder: str | Path,
    job_base: str,
    *,
    single_site: bool | None = None,
    has_aoe: bool | None = None,
) -> dict:
    """Compute per-detector efficiencies from a Polars lazy scan.

    For every combination of simulated energy and detector listed in
    *eres_dict*, the function:

    1. Looks up the detector-specific FWHM from *eres_dict*.
    2. Defines the full-energy-peak integration window as
       ``[e_value - 2·FWHM, e_value + 2·FWHM]``, where ``e_value`` is the
       reconstructed energy corresponding to the simulated energy ``ene``.
    3. Collects the lazy frame *lf* **once per energy** and uses a
       vectorised join + group-by to count events inside the FEP window
       for every detector simultaneously.
    4. Reads ``n_primaries`` from the STP files for each detector.
    5. Computes ``ratio = n_events / n_primaries``
       (or ``nan`` when ``n_primaries == 0``).

    Parameters
    ----------
    lf
        A Polars *lazy* scan of the parquet dataset partitioned by ``sim_e``.
        Expected columns: ``rawid``, ``energy``, ``sim_e``, ``is_good_channel``.
    eres_dict
        Nested dictionary with structure
        ``{energy: {detector_name: {"fwhm": float, ...}, ...}, ...}``
        as produced by the resolution-extraction pipeline
        (e.g. ``eres_per_det_tot.yaml``).
    simulated_energies
        List of simulated energies (in keV) to iterate over. The ``ene``
        in the loop corresponds to that simulated-energy partition.
    chmap
        LEGEND channel-map object (``LegendMetadata.channelmap(...)``).
        Must support ``chmap[det_name].daq.rawid``.
    scratch_folder
        Base scratch directory containing the generated LH5 STP files.
    job_base
        Job string template containing the ``{ene}`` placeholder.
    single_site
        If True, keep only rows where ``has_aoe`` and ``is_single_site`` are True.
        If False, keep only rows where ``has_aoe`` is True and ``is_single_site`` is False.
        If None (default), no filtering is applied.
    has_aoe
        If True, keep only rows where ``has_aoe`` is True.
        If None (default), no filtering is applied. Automatically set to True if ``single_site`` is used.

    Returns
    -------
    dict
        Nested dictionary with structure::

            {
                energy: {
                    detector_name: {
                        "n_events":         int,
                        "n_primaries":      int,
                        "ratio":            float,
                        "ratio_sigma":      float,
                        "ratio_sigma_freq": float,
                        "ratio_syst_fwhm":  float,
                        "expo":             float,   # from eres_dict
                    },
                    ...
                },
                ...
            }
    """
    if single_site is not None:
        has_aoe = True

    ratio_dict: dict = {}

    for ene in tqdm(simulated_energies):
        ene_key = int(ene)
        ratio_dict[ene_key] = {}

        # Skip energies not present in the resolution dictionary
        if ene_key not in eres_dict:
            logger.warning("Energy %d keV not found in eres_dict, skipping", ene_key)
            continue

        det_eres = eres_dict[ene_key]

        # Format the job string and find STP files once per energy
        job_string = job_base.format(ene=ene)
        stp_files = [
            str(p)
            for p in Path(f"{scratch_folder}/generated/tier/stp/{job_string}/").glob(
                f"l200cfg01-{job_string}-job_*-tier_stp.lh5"
            )
        ]

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

            n_primaries = 0
            for stp_file in stp_files:
                stp_ge = lh5.read_as(f"/stp/{det_name}", stp_file, library="ak")
                n_primaries += len(np.unique(ak.to_numpy(stp_ge.evtid)))
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
                    "low": ene - 2.0 * fwhm,
                    "high": ene + 2.0 * fwhm,
                    "low_up": ene - 2.0 * fwhm_up,
                    "high_up": ene + 2.0 * fwhm_up,
                    "low_down": ene - 2.0 * fwhm_down,
                    "high_down": ene + 2.0 * fwhm_down,
                }
            )

        if not det_rows:
            continue

        det_df = pl.DataFrame(det_rows)
        known_rawids = det_df["rawid"].to_list()

        # --- Phase 2: single collect per energy -----------------------------
        filter_expr = (
            (pl.col("is_good_channel"))
            & (pl.col("sim_e") == ene)
            & (pl.col("rawid").is_in(known_rawids))
        )
        if has_aoe is True:
            filter_expr &= pl.col("has_aoe")

        if single_site is True:
            filter_expr &= pl.col("is_single_site")
        elif single_site is False:
            filter_expr &= ~pl.col("is_single_site")

        df_ene = lf.filter(filter_expr).select("rawid", "energy").collect()

        # --- Phase 3: vectorised FEP counting via join + group_by -----------
        df_joined = df_ene.join(det_df, on="rawid")

        counts = df_joined.group_by("det_name").agg(
            pl.col("energy")
            .is_between(pl.col("low"), pl.col("high"))
            .sum()
            .alias("n_events"),
            pl.col("energy")
            .is_between(pl.col("low_up"), pl.col("high_up"))
            .sum()
            .alias("n_events_up"),
            pl.col("energy")
            .is_between(pl.col("low_down"), pl.col("high_down"))
            .sum()
            .alias("n_events_down"),
        )

        # Index counts by detector name for fast lookup
        counts_map = {row["det_name"]: row for row in counts.iter_rows(named=True)}

        # --- Phase 4: compute efficiencies ----------------------------------
        for det_name, n_primaries in n_prim_map.items():
            row = counts_map.get(det_name)

            # When has_aoe filtering is active, detectors with no surviving
            # events lack AoE information entirely — set everything to zero.
            if has_aoe is True and row is None:
                ratio_dict[ene_key][det_name] = {
                    "n_events": 0,
                    "n_primaries": n_primaries,
                    "ratio": 0.0,
                    "ratio_sigma": 0.0,
                    "ratio_sigma_freq": 0.0,
                    "ratio_syst_fwhm": 0.0,
                    "expo": expo_map[det_name],
                }
                continue

            n_events = row["n_events"] if row else 0
            n_events_up = row["n_events_up"] if row else 0
            n_events_down = row["n_events_down"] if row else 0

            if n_primaries > 0:
                eff = n_events / n_primaries
                ratio_sigma_freq = float(np.sqrt(eff * (1.0 - eff) / n_primaries))
            else:
                ratio_sigma_freq = 0.0

            ratio, ratio_sigma = bayesian_efficiency(n_events, n_primaries)
            ratio_up, _ = bayesian_efficiency(n_events_up, n_primaries)
            ratio_down, _ = bayesian_efficiency(n_events_down, n_primaries)

            ratio_syst_fwhm = abs(ratio_up - ratio_down) / 2.0

            ratio_dict[ene_key][det_name] = {
                "n_events": n_events,
                "n_primaries": n_primaries,
                "ratio": ratio,
                "ratio_sigma": ratio_sigma,
                "ratio_sigma_freq": ratio_sigma_freq,
                "ratio_syst_fwhm": ratio_syst_fwhm,
                "expo": expo_map[det_name],
            }

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


_FEP_COLORS = {
    "e$^-$ + $\\gamma$": "red",
    "$\\gamma$": "green",
    "e$^-$": "purple",
}


def plot_lar_cut_spectra(
    lf: pl.LazyFrame,
    simulated_energies: Sequence[int],
    chmap: object,
    bin_factor: int = 2,
    x_range: tuple[float, float] | None = None,
    save_dir: str | Path = "notebooks/plots",
) -> None:
    """Plot the LAr-veto survival fraction as a function of energy.

    For each simulated energy a 2x2 figure is produced with one panel
    per detector type (BEGe, ICPC, PPC, COAX).  Each panel shows the
    bin-by-bin survival fraction ``SF(E) = N_surviving / N_total``
    with Bayesian uncertainty bands (Beta conjugate prior,
    ``Beta(0.5, 0.5)``).  Three vertical bands mark the expected
    full-energy-peak positions of the dark-Compton process:
    e⁻ + gamma (total), gamma only, and e⁻ only.

    Parameters
    ----------
    lf
        Polars lazy scan of the parquet dataset.
        Expected columns: ``rawid``, ``energy``, ``sim_e``,
        ``is_good_channel``, ``coincident_spms``.
    simulated_energies
        Simulated energies (keV) to iterate over.
    chmap
        LEGEND channel-map object.  Used to map ``rawid`` → detector
        type via ``chmap.map("daq.rawid")``.
    # bins
    #     Number of histogram bins (default 200).
    x_range
        Optional ``(low, high)`` tuple for the x-axis range.  If
        *None*, the range is determined from the data.
    save_dir
        Directory where figures are saved.  Figures are named
        ``lar_survival_fraction_{ene}keV.png``.
    """
    from dark_compton_generators import calculate_energies  # noqa: PLC0415

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    det_type_order = ["BEGe", "ICPC", "PPC", "COAX"]

    for ene in tqdm(simulated_energies):
        # --- Collect data for this energy -----------------------------------
        df = (
            lf.filter(pl.col("is_good_channel") & (pl.col("sim_e") == ene))
            .select("rawid", "energy", "coincident_spms")
            .collect()
        )

        if df.is_empty():
            logger.warning("No data for %d keV, skipping", ene)
            continue

        # --- Map rawid → detector type via channel map ----------------------
        rawid_map = chmap.map("daq.rawid")
        unique_rawids = df["rawid"].unique().to_list()

        type_rows: list[dict] = []
        for rid in unique_rawids:
            try:
                name = rawid_map[rid]["name"]
                det_type = _DET_TYPE_MAP.get(name[0].upper())
                if det_type is not None:
                    type_rows.append({"rawid": rid, "det_type": det_type})
            except (KeyError, IndexError):
                continue

        if not type_rows:
            logger.warning("No mappable rawids for %d keV, skipping", ene)
            continue

        type_df = pl.DataFrame(type_rows)
        df = df.join(type_df, on="rawid", how="inner")

        # --- Determine common x-range --------------------------------------
        xlim = (
            x_range
            if x_range is not None
            else (
                float(df["energy"].min()),
                float(df["energy"].max()),
            )
        )

        # --- FEP positions for this simulated energy ------------------------
        e_elec, e_gamma = calculate_energies(ene)
        fep_lines = {
            "e$^-$ + $\\gamma$": float(e_elec + e_gamma),
            "$\\gamma$": float(e_gamma),
            "e$^-$": float(e_elec),
        }

        # --- Plot 2x2 figure ------------------------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"LAr veto survival fraction - simulated {ene} keV",
            fontsize=16,
        )

        for ax, det_type in zip(axes.flat, det_type_order, strict=False):
            subset = df.filter(pl.col("det_type") == det_type)

            if subset.is_empty():
                ax.set_title(det_type, fontsize=13)
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="grey",
                )
                continue

            all_energy = subset["energy"].to_numpy()
            surv_energy = subset.filter(~pl.col("coincident_spms"))["energy"].to_numpy()

            # Bin-by-bin counts
            n_total, bin_edges = np.histogram(
                all_energy,
                bins=int(ene / bin_factor),
                range=xlim,
            )
            n_surv, _ = np.histogram(surv_energy, bins=bin_edges)
            bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            # Bayesian survival fraction per bin
            mask = n_total > 0
            sf = np.full_like(n_total, np.nan, dtype=float)
            sf_sigma = np.full_like(n_total, np.nan, dtype=float)

            for i in np.where(mask)[0]:
                sf[i], sf_sigma[i] = bayesian_efficiency(
                    int(n_surv[i]),
                    int(n_total[i]),
                )

            ax.errorbar(
                bin_centres[mask],
                sf[mask],
                yerr=sf_sigma[mask],
                fmt=".",
                markersize=3,
                linewidth=0.8,
                color=_DET_TYPE_COLOR[det_type],
            )

            # --- Shade the 3 FEP regions ------------------------------------
            bin_width = bin_edges[1] - bin_edges[0]
            for label, e_fep in fep_lines.items():
                ax.axvspan(
                    e_fep - bin_width,
                    e_fep + bin_width,
                    alpha=0.20,
                    color=_FEP_COLORS[label],
                    label=label,
                )

            ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.6)
            ax.set_ylim(-0.05, 1.15)
            ax.set_xlabel("Energy in HPGe [keV]", fontsize=12)
            ax.set_ylabel("Survival fraction", fontsize=12)
            ax.set_title(det_type, fontsize=13)
            ax.legend(fontsize=8, loc="lower left")

        fig.tight_layout()
        fig.savefig(
            save_path / f"lar_survival_fraction_{ene}keV.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()


def plot_aoe_by_detector(
    lf: pl.LazyFrame,
    eres_dict: Mapping,
    chmap: object,
    sim_e: int,
    title_string: str | None = None,
    energy_range: tuple[float, float] | None = None,
    save_plot: bool = True,
    show_plot: bool = True,
    plot_m1: bool = True,
    bin_edges: Sequence[float] | int = 150,
    save_dir: str | Path = "plots",
) -> None:
    """Plot AoE histograms for each detector, split by coincident_spms status.

    Only events with energy inside the specified energy range (or sim_e +- 2*FWHM
    if not specified) are plotted. Two histograms are plotted: one for coincident_spms
    == True (M1, step style, conditional on plot_m1) and one for coincident_spms
    == False (M1 + LAr, filled style with alpha = 0.4).

    Parameters
    ----------
    lf
        Polars LazyFrame containing the data.
    eres_dict
        Dictionary containing energy resolution (FWHM) per detector.
    chmap
        LEGEND channel-map object.
    sim_e
        Selected simulated energy in keV.
    energy_range
        Optional custom energy range (low, high) in keV to study. If None, the FWHM
        window (sim_e +- 2*FWHM) is computed per detector.
    save_plot
        Whether to save the generated plots to files.
    show_plot
        Whether to display the generated plots.
    plot_m1
        Whether to plot the M1 histogram (coincident_spms == True).
    bin_edges
        Number of histogram bins (int) or sequence defining the bin edges.
    save_dir
        Directory where the generated plots will be saved if save_plot is True.
    """
    ene_key = int(sim_e)
    if ene_key not in eres_dict:
        msg = f"Energy {ene_key} not found in eres_dict."
        raise KeyError(msg)

    det_eres = eres_dict[ene_key]

    # --- Phase 1: Build per-detector info ----------
    det_rows = []
    for det_name, eres_info in det_eres.items():
        try:
            rawid = chmap[det_name].daq.rawid
        except (KeyError, AttributeError):
            logger.warning("Cannot resolve rawid for %s, skipping", det_name)
            continue

        if energy_range is not None:
            low = energy_range[0]
            high = energy_range[1]
        else:
            fwhm = float(eres_info["fwhm"])
            low = sim_e - 2.0 * fwhm
            high = sim_e + 2.0 * fwhm

        det_rows.append(
            {
                "rawid": rawid,
                "det_name": det_name,
                "low": low,
                "high": high,
            }
        )

    if not det_rows:
        logger.warning("No detector info found for simulated energy %d keV", sim_e)
        return

    det_df = pl.DataFrame(det_rows)
    known_rawids = det_df["rawid"].to_list()

    # --- Phase 2: Collect data for the selected energy ----------
    df_ene = (
        lf.filter(
            (pl.col("is_good_channel"))
            & (pl.col("sim_e") == sim_e)
            & (pl.col("rawid").is_in(known_rawids))
        )
        .select("rawid", "energy", "aoe", "coincident_spms")
        .collect()
    )

    if df_ene.is_empty():
        logger.warning("No data found for simulated energy %d keV", sim_e)
        return

    # --- Phase 3: Join and filter within FWHM or custom energy window ----------
    df_joined = df_ene.join(det_df, on="rawid")
    df_filtered = df_joined.filter(
        pl.col("energy").is_between(pl.col("low"), pl.col("high"))
    )

    if df_filtered.is_empty():
        logger.warning("No events within window for simulated energy %d keV", sim_e)
        return

    # --- Phase 4: Plot per detector ----------
    if save_plot:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if energy_range is not None:
            pdf_name = f"aoe_hist_all_dets_{energy_range[0]:0.0f}_{energy_range[1]:0.0f}keV.pdf"
        else:
            pdf_name = f"aoe_hist_all_dets_{sim_e}keV.pdf"

        pdf_pages = PdfPages(save_path / pdf_name)

    unique_dets = df_filtered["det_name"].unique().to_list()

    for det_name in unique_dets:
        det_df = df_filtered.filter(pl.col("det_name") == det_name)
        if det_df.is_empty():
            continue

        aoe_anticoincident = det_df.filter(~pl.col("coincident_spms"))["aoe"].to_numpy()
        aoe_anticoincident_clean = aoe_anticoincident[np.isfinite(aoe_anticoincident)]

        if plot_m1:
            aoe_coincident = det_df["aoe"].to_numpy()
            aoe_coincident_clean = aoe_coincident[np.isfinite(aoe_coincident)]
            aoe_all_clean = aoe_coincident_clean
            # Count events and NaNs for M1 (commented out as requested)
            # n_coincident = len(aoe_coincident)
            # nans_coincident = int(np.isnan(aoe_coincident).sum())
        else:
            aoe_all_clean = np.array([])

        # Count events and NaNs for anticoincident (commented out as requested)
        # n_anticoincident = len(aoe_anticoincident)
        # nans_anticoincident = int(np.isnan(aoe_anticoincident).sum())

        aoe_all = np.concatenate([aoe_all_clean, aoe_anticoincident_clean])
        if len(aoe_all) == 0:
            continue

        if isinstance(bin_edges, (int, str)):
            xmin, xmax = np.percentile(aoe_all, [0.1, 99.9])
            hist_range = None if xmin == xmax else (xmin, xmax)
        else:
            hist_range = None

        plt.figure(figsize=(10, 6))

        # M1 : step style
        if plot_m1:
            plt.hist(
                aoe_coincident_clean,
                bins=bin_edges,
                range=hist_range,
                label="M1",
                histtype="step",
                linewidth=1.5,
                color="tab:blue",
            )

        # M1 + LAr cut (coincident_spms == False): filled but with alpha = 0.4
        plt.hist(
            aoe_anticoincident_clean,
            bins=bin_edges,
            range=hist_range,
            label="M1 + LAr",
            histtype="stepfilled",
            alpha=0.4,
            color="tab:orange",
        )

        plt.xlabel("AoE [a.u.]", fontsize=12)
        plt.ylabel("Counts", fontsize=12)

        if energy_range is not None:
            title_str = f"AoE distribution - {det_name} ({energy_range[0]:0.0f} - {energy_range[1]:0.0f} keV)"
        else:
            fwhm = float(det_eres[det_name]["fwhm"])
            title_str = f"AoE distribution - {det_name} ({sim_e}+-{2 * fwhm:0.1f} keV)"

        plt.yscale("log")

        plt.title(title_str, fontsize=13)
        # plt.grid(True, linestyle="--", alpha=0.5)
        if title_string is not None:
            plt.legend(fontsize=10, title=title_string)
        else:
            plt.legend(fontsize=10)

        plt.tight_layout()
        if save_plot:
            pdf_pages.savefig(bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close()

    if save_plot:
        pdf_pages.close()


def plot_aoe_by_detector_type(
    lf: pl.LazyFrame,
    eres_dict: Mapping,
    chmap: object,
    simulated_energies: Sequence[int],
    title_string: str | None = None,
    save_plot: bool = True,
    show_plot: bool = True,
    bin_edges: Sequence[float] | int = 150,
    save_dir: str | Path = "plots",
) -> None:
    """Plot AoE histograms grouped by detector type for each simulated energy.

    For each simulated energy, a single figure is produced with all detector
    types (BEGe, ICPC, PPC, COAX) overlaid as step histograms. Only events
    within each detector's specific FWHM window (sim_e ± 2·FWHM) are plotted,
    and only for anticoincident (M1 + LAr cut) events.

    Parameters
    ----------
    lf
        Polars LazyFrame containing the data.
    eres_dict
        Nested dictionary ``{energy: {det_name: {"fwhm": float, ...}, ...}}``.
    chmap
        LEGEND channel-map object.
    simulated_energies
        List of simulated energies in keV to iterate over.
    title_string
        Optional legend title string.
    save_plot
        Whether to save the generated plots to files.
    show_plot
        Whether to display the generated plots.
    bin_edges
        Number of histogram bins (int) or sequence defining the bin edges.
    save_dir
        Directory where the generated plots will be saved if *save_plot* is
        True.
    """
    if save_plot:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

    det_types = list(_DET_TYPE_MAP.values())  # BEGe, COAX, ICPC, PPC

    for sim_e in tqdm(simulated_energies):
        ene_key = int(sim_e)
        if ene_key not in eres_dict:
            logger.warning("Energy %d keV not found in eres_dict, skipping", ene_key)
            continue

        det_eres = eres_dict[ene_key]

        # --- Phase 1: Build per-detector info with detector type ----------
        det_rows: list[dict] = []
        for det_name, eres_info in det_eres.items():
            try:
                rawid = chmap[det_name].daq.rawid
            except (KeyError, AttributeError):
                logger.warning("Cannot resolve rawid for %s, skipping", det_name)
                continue

            prefix = det_name[0].upper()
            det_type = _DET_TYPE_MAP.get(prefix)
            if det_type is None:
                continue

            fwhm = float(eres_info["fwhm"])
            low = sim_e - 2.0 * fwhm
            high = sim_e + 2.0 * fwhm

            det_rows.append(
                {
                    "rawid": rawid,
                    "det_name": det_name,
                    "det_type": det_type,
                    "low": low,
                    "high": high,
                }
            )

        if not det_rows:
            logger.warning("No detector info found for simulated energy %d keV", sim_e)
            continue

        det_df = pl.DataFrame(det_rows)
        known_rawids = det_df["rawid"].to_list()

        # --- Phase 2: Collect data for the selected energy ----------
        df_ene = (
            lf.filter(
                (pl.col("is_good_channel"))
                & (pl.col("sim_e") == sim_e)
                & (pl.col("rawid").is_in(known_rawids))
            )
            .select("rawid", "energy", "aoe", "coincident_spms")
            .collect()
        )

        if df_ene.is_empty():
            logger.warning("No data found for simulated energy %d keV", sim_e)
            continue

        # --- Phase 3: Join and filter within energy window ----------
        df_joined = df_ene.join(det_df, on="rawid")
        df_filtered = df_joined.filter(
            pl.col("energy").is_between(pl.col("low"), pl.col("high"))
        )

        if df_filtered.is_empty():
            logger.warning("No events within window for simulated energy %d keV", sim_e)
            continue

        # --- Phase 4: single figure, one histogram per detector type ----------
        fig, ax = plt.subplots(figsize=(10, 6))

        # Compute shared bin range across all detector types
        aoe_all_types = df_filtered.filter(
            pl.col("coincident_spms") == False  # noqa: E712
        )["aoe"].to_numpy()
        aoe_all_types = aoe_all_types[np.isfinite(aoe_all_types)]

        if len(aoe_all_types) == 0:
            logger.warning("No finite AoE values for simulated energy %d keV", sim_e)
            plt.close(fig)
            continue

        if isinstance(bin_edges, int):
            xmin, xmax = np.percentile(aoe_all_types, [0.1, 99.9])
            hist_range = None if xmin == xmax else (xmin, xmax)
        else:
            hist_range = None

        for det_type in det_types:
            det_type_df = df_filtered.filter(
                (pl.col("det_type") == det_type) & (pl.col("coincident_spms") == False)  # noqa: E712
            )
            if det_type_df.is_empty():
                continue

            aoe = det_type_df["aoe"].to_numpy()
            aoe_clean = aoe[np.isfinite(aoe)]
            if len(aoe_clean) == 0:
                continue

            ax.hist(
                aoe_clean,
                bins=bin_edges,
                range=hist_range,
                label=det_type,
                histtype="step",
                linewidth=1.5,
                color=_DET_TYPE_COLOR[det_type],
            )

        ax.set_xlabel("AoE [a.u.]", fontsize=12)
        ax.set_ylabel("Counts", fontsize=12)
        ax.set_yscale("log")

        title_str = f"AoE by detector type (M1 + LAr) — {sim_e} keV (±2·FWHM window)"
        save_name = f"aoe_det_type_{sim_e}keV.png"

        ax.set_title(title_str, fontsize=13)

        if title_string is not None:
            ax.legend(fontsize=10, title=title_string)
        else:
            ax.legend(fontsize=10)

        fig.tight_layout()

        if save_plot:
            plot_file = save_path / save_name
            fig.savefig(plot_file, dpi=300, bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close(fig)


def plot_aoe_vs_energy_2d(
    lf: pl.LazyFrame,
    chmap: object = None,
    group_by: str | None = None,
    save_plot: bool = True,
    show_plot: bool = True,
    bins: tuple[int, int] | int = (150, 150),
    energy_range: tuple[float, float] | None = None,
    aoe_range: tuple[float, float] | None = None,
    title_string: str | None = None,
    filename_string: str | None = None,
    save_dir: str | Path = "plots",
) -> None:
    """Plot 2D histogram of AoE vs Energy.

    Filters by `has_aoe` == True before plotting.

    Parameters
    ----------
    lf
        Polars LazyFrame containing the data.
    chmap
        LEGEND channel-map object. Required if group_by is not None.
    group_by
        How to group the plots. Valid options: None (all together), "det_type", or "det_name".
    save_plot
        Whether to save the generated plots to files.
    show_plot
        Whether to display the generated plots.
    bins
        Number of histogram bins (int or tuple of ints).
    energy_range
        Optional range for the energy (x) axis.
    aoe_range
        Optional range for the AoE (y) axis.
    title_string
        Optional string to prepend to the plot title.
    filename_string
        Optional string to append to the plot filename.
    save_dir
        Directory where the generated plots will be saved if save_plot is True.
    """
    if save_plot:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        pdf_name = f"aoe_vs_energy_2d{'_' + group_by if group_by else ''}{'_' + filename_string if filename_string else ''}.pdf"
        pdf_pages = PdfPages(save_path / pdf_name)

    lf_valid = lf.filter(pl.col("has_aoe"))

    if energy_range is not None:
        lf_valid = lf_valid.filter(pl.col("energy").is_between(*energy_range))
    if aoe_range is not None:
        lf_valid = lf_valid.filter(pl.col("aoe").is_between(*aoe_range))

    def _get_h_range(energy_arr, aoe_arr):
        if energy_range is not None and aoe_range is not None:
            return [energy_range, aoe_range]
        if energy_range is not None or aoe_range is not None:
            xmin, xmax = (
                energy_range
                if energy_range is not None
                else (np.nanmin(energy_arr), np.nanmax(energy_arr))
            )
            ymin, ymax = (
                aoe_range
                if aoe_range is not None
                else (np.nanmin(aoe_arr), np.nanmax(aoe_arr))
            )
            if np.isnan(xmin):
                xmin, xmax = 0, 1
            if np.isnan(ymin):
                ymin, ymax = 0, 1
            return [[xmin, xmax], [ymin, ymax]]
        return None

    if group_by is None:
        df = lf_valid.select("energy", "aoe").collect()
        if df.is_empty():
            logger.warning("No data found for AoE vs Energy plot")
            if save_plot:
                pdf_pages.close()
            return

        energy = df["energy"].to_numpy()
        aoe = df["aoe"].to_numpy()

        plt.figure(figsize=(10, 6))

        plt.hist2d(
            energy,
            aoe,
            bins=bins,
            range=_get_h_range(energy, aoe),
            cmap="viridis",
            cmin=1,
            norm=LogNorm(),
        )
        plt.colorbar(label="Counts")
        plt.xlabel("Energy [keV]", fontsize=12)
        plt.ylabel("AoE [a.u.]", fontsize=12)
        base_title = "AoE vs Energy (All Detectors)"
        plt.title(
            f"{base_title} - {title_string}" if title_string else base_title,
            fontsize=13,
        )
        plt.tight_layout()

        if save_plot:
            pdf_pages.savefig(bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close()

    elif group_by in ("det_type", "det_name"):
        if chmap is None:
            logger.error("chmap must be provided when group_by is not None.")
            if save_plot:
                pdf_pages.close()
            return

        unique_rawids = (
            lf_valid.select("rawid").unique().collect().drop_nulls()["rawid"].to_list()
        )

        groups = {}
        if group_by == "det_type":
            groups = get_rawid_lists(chmap, unique_rawids)
        elif group_by == "det_name":
            for rid in unique_rawids:
                try:
                    ge = chmap.map("daq.rawid")[rid]["name"]
                    groups[ge] = [rid]
                except KeyError:
                    pass

        for g, rawids_list in groups.items():
            if not rawids_list:
                continue

            df = (
                lf_valid.filter(pl.col("rawid").is_in(rawids_list))
                .select("energy", "aoe")
                .collect()
            )
            if df.is_empty():
                continue

            energy = df["energy"].to_numpy()
            aoe = df["aoe"].to_numpy()

            plt.figure(figsize=(10, 6))

            plt.hist2d(
                energy,
                aoe,
                bins=bins,
                range=_get_h_range(energy, aoe),
                cmap="viridis",
                cmin=1,
                norm=LogNorm(),
            )
            plt.colorbar(label="Counts")
            plt.xlabel("Energy [keV]", fontsize=12)
            plt.ylabel("AoE [a.u.]", fontsize=12)
            base_title = f"AoE vs Energy - {g}"
            plt.title(
                f"{base_title} - {title_string}" if title_string else base_title,
                fontsize=13,
            )
            plt.tight_layout()

            if save_plot:
                pdf_pages.savefig(bbox_inches="tight")
            if show_plot:
                plt.show()
            plt.close()
    else:
        logger.error(
            "Invalid group_by: %s. Use None, 'det_type', or 'det_name'.", group_by
        )

    if save_plot:
        pdf_pages.close()

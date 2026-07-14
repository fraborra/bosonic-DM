# Copyright (C) 2025 Francesco Borra
#

"""Energy resolution extraction and FWHM computation for bosonic-DM analysis."""

from __future__ import annotations

import functools
import logging

import numpy as np
from dbetto import Props
from legendmeta import LegendMetadata
from tqdm import tqdm

from bosonic_dm.plotting.utils import _DET_TYPE_MAP
from bosonic_dm.stats import compute_weighted_uncertainty, weighted_mean

logger = logging.getLogger(__name__)


def weighted_resolution_from_nested_dict(data: dict, E: float) -> float:
    """Compute exposure-weighted FWHM resolution sqrt(a + b*E) from a nested dict structure.

    Parameters
    ----------
    data : dict
        Nested dict like {period: {run: {detector: {"expo", "a", "b", "a_unc", "b_unc", "ab_corr"}}}}.
    E : float
        Energy at which to compute the resolution.

    Returns
    -------
    f_weighted : float
        Weighted mean FWHM at energy E.
    """
    a_list, b_list, w_list = [], [], []
    a_unc_list, b_unc_list, cab_list = [], [], []

    for period, period_dict in data.items():
        for run, run_dict in period_dict.items():
            for _channel, vals in run_dict.items():
                if vals["usability"] != "on":
                    continue
                # skip missing keys
                if "a" not in vals or "b" not in vals or "expo" not in vals:
                    logger.info(
                        f"...found no values for retrieving FWHM in {period}-{run}, skip it"
                    )
                    continue

                a_val, b_val, w_val = vals["a"], vals["b"], vals["expo"]

                a_list.append(a_val)
                b_list.append(b_val)
                w_list.append(w_val)
                a_unc_list.append(vals.get("a_unc", 0.0))
                b_unc_list.append(vals.get("b_unc", 0.0))
                cab_list.append(vals.get("ab_corr", 0.0))

    # arrays
    a_arr = np.array(a_list)
    b_arr = np.array(b_list)
    w_arr = np.array(w_list)

    a_unc_arr = np.array(a_unc_list)
    b_unc_arr = np.array(b_unc_list)
    cab_arr = np.array(cab_list)

    # per-detector FWHM
    fwhm_arr = np.sqrt(a_arr + b_arr * E)

    # propagate uncertainties
    sigma_f_arr = propagate_resolution_uncertainty(
        a_arr, b_arr, a_unc_arr, b_unc_arr, cab_arr, E
    )

    # weighted average
    # total_expo = np.sum(w_arr)
    fwhm_weighted = weighted_mean(fwhm_arr, w_arr)

    # total uncertainty
    unc_total = compute_weighted_uncertainty(
        w_arr, fwhm_arr, fwhm_weighted, sigma_f_arr
    )

    return fwhm_weighted, unc_total


def compute_fwhm(a: float, b: float, energy: float) -> float:
    """Compute FWHM resolution sqrt(a + b * energy).

    Parameters
    ----------
    a : float
        Constant term of the energy resolution parametrisation.
    b : float
        Linear term of the energy resolution parametrisation.
    energy : float
        Energy at which to evaluate the FWHM.

    Returns
    -------
    float
        FWHM value at the given energy.
    """
    return np.sqrt(a + b * energy)


def propagate_resolution_uncertainty(
    a: float | np.ndarray,
    b: float | np.ndarray,
    a_unc: float | np.ndarray,
    b_unc: float | np.ndarray,
    ab_corr: float | np.ndarray,
    energy: float | np.ndarray,
) -> float | np.ndarray:
    """Propagate uncertainties of a and b to the FWHM resolution sqrt(a + b * energy).

    Parameters
    ----------
    a : float or np.ndarray
        Constant term of the resolution parametrisation.
    b : float or np.ndarray
        Linear term of the resolution parametrisation.
    a_unc : float or np.ndarray
        Uncertainty on a.
    b_unc : float or np.ndarray
        Uncertainty on b.
    ab_corr : float or np.ndarray
        Covariance (or correlation-derived covariance) between a and b.
    energy : float or np.ndarray
        Energy at which to evaluate the uncertainty.

    Returns
    -------
    float or np.ndarray
        Propagated uncertainty on the FWHM.
    """
    fwhm = compute_fwhm(a, b, energy)
    df_da = 1.0 / (2.0 * fwhm)
    df_db = energy / (2.0 * fwhm)

    return np.sqrt(
        (df_da * a_unc) ** 2 + (df_db * b_unc) ** 2 + 2 * df_da * df_db * ab_corr
    )


def weighted_resolution_per_detector(data: dict, E: float) -> dict:
    """Compute exposure-weighted FWHM resolution sqrt(a + b*E) per detector.

    Averaging over all period-run combinations.

    Parameters
    ----------
    data : dict
        Nested dict like {period: {run: {detector: {"expo", "a", "b", "a_unc", "b_unc", "ab_corr"}}}}.
    E : float
        Energy at which to compute the resolution.

    Returns
    -------
    result : dict
        Dict like {detector: { 'fwhm': fwhm_weighted, 'expo': total_expo}}.
    """
    # collect per-detector lists across all period-run
    det_data = {}

    for period, period_dict in data.items():
        for run, run_dict in period_dict.items():
            for det, vals in run_dict.items():
                if vals.get("usability") != "on":
                    continue
                if "a" not in vals or "b" not in vals or "expo" not in vals:
                    logger.info(
                        f"...found no values for retrieving FWHM in {period}-{run}-{det}, skip it"
                    )
                    continue

                if det not in det_data:
                    det_data[det] = {
                        "a": [],
                        "b": [],
                        "w": [],
                        "sa": [],
                        "sb": [],
                        "cab": [],
                    }

                det_data[det]["a"].append(vals["a"])
                det_data[det]["b"].append(vals["b"])
                det_data[det]["w"].append(vals["expo"])
                det_data[det]["sa"].append(vals.get("a_unc", 0.0))
                det_data[det]["sb"].append(vals.get("b_unc", 0.0))
                det_data[det]["cab"].append(vals.get("ab_corr", 0.0))

    result = {}

    for det, d in det_data.items():
        a_arr = np.array(d["a"])
        b_arr = np.array(d["b"])
        w_arr = np.array(d["w"])
        sa_arr = np.array(d["sa"])
        sb_arr = np.array(d["sb"])
        cab_arr = np.array(d["cab"])

        # per-run sigma
        fwhm_arr = compute_fwhm(a_arr, b_arr, E)

        # propagate uncertainties
        sigma_f_arr = propagate_resolution_uncertainty(
            a_arr, b_arr, sa_arr, sb_arr, cab_arr, E
        )

        # weighted average
        total_expo = np.sum(w_arr)
        fwhm_weighted = weighted_mean(fwhm_arr, w_arr)

        # total uncertainty
        unc_total = compute_weighted_uncertainty(
            w_arr, fwhm_arr, fwhm_weighted, sigma_f_arr
        )

        result[det] = {"fwhm": fwhm_weighted, "unc": unc_total, "expo": total_expo}

    return result


def weighted_resolution_per_detector_type(data: dict) -> dict:
    """Compute exposure-weighted FWHM and total exposure per detector type.

    Thin wrapper around :func:`weighted_value_per_det_type` that accepts the
    flat ``{detector: {"fwhm": ..., "unc": ..., "expo": ...}}`` output of
    :func:`weighted_resolution_per_detector` and groups results by detector
    type.  Exposure is **summed** (not averaged) across detectors of the same
    type.

    Parameters
    ----------
    data : dict
        Flat dict ``{detector: {"fwhm": float, "unc": float, "expo": float}}`` as returned
        by :func:`weighted_resolution_per_detector`.

    Returns
    -------
    result : dict
        ``{det_type: {"fwhm": weighted_mean_fwhm, "unc": unc_total, "expo": total_expo}}`` for
        each detector type that has at least one valid entry.
    """
    # Wrap the flat dict in a fake period/run level so that
    # weighted_value_per_det_type can traverse it uniformly.
    wrapped = {"p00": {"r000": data}}
    fwhm_by_type = weighted_value_per_det_type(
        wrapped,
        value_keys="fwhm",
        weight_key="expo",
        unc_key="unc",
        usability_filter=None,
        fill_missing=None,
    )

    # Sum (not weight-average) exposure per detector type.
    expo_by_type: dict[str, float] = dict.fromkeys(_DET_TYPE_MAP.values(), 0.0)
    for det, d in data.items():
        prefix = det[0].upper()
        det_type = _DET_TYPE_MAP.get(prefix)
        if det_type is None:
            logger.warning("Unknown detector prefix '%s' for detector %s", prefix, det)
            continue
        expo_by_type[det_type] += d.get("expo", 0.0)

    result = {}
    for det_type in _DET_TYPE_MAP.values():
        fwhm_data = fwhm_by_type.get(det_type, {}).get("fwhm")
        if fwhm_data is None:
            continue

        fwhm = fwhm_data.get("value", np.nan)
        unc = fwhm_data.get("unc", np.nan)
        expo = expo_by_type[det_type]
        if np.isfinite(fwhm) and expo > 0:
            result[det_type] = {"fwhm": fwhm, "unc": unc, "expo": expo}

    return result


def weighted_value_per_det_type(
    data: dict,
    value_keys: str | list[str],
    *,
    weight_key: str = "expo",
    unc_key: str | None = None,
    usability_filter: str | None = "on",
    fill_missing: float | None = None,
) -> dict:
    """Compute exposure-weighted mean of one or more scalar keys, grouped by detector type.

    Traverses ``{period: {run: {detector: {weight_key, value_key, ...}}}}`` and
    returns the weighted average ``sum(w * v) / sum(w)`` for each detector type
    (BEGe, COAX, ICPC, PPC), identified by the first letter of the detector name.

    Parameters
    ----------
    data : dict
        Nested dict with structure
        ``{period: {run: {detector: {weight_key: ..., value_key: ..., ...}}}}``.
    value_keys : str or list of str
        Key(s) of the scalar value(s) to average inside the innermost dict.
    weight_key : str, optional
        Key used as the exposure weight. Defaults to ``"expo"``.
    unc_key : str or None, optional
        If provided, the key in the innermost dict that holds the pre-computed
        uncertainty on each entry.
    usability_filter : str or None, optional
        If not ``None``, only detector entries whose ``"usability"`` field
        equals this value are included. Defaults to ``"on"``.
    fill_missing : float or None, optional
        Value to use for detector types with no valid data.

    Returns
    -------
    result : dict
        Weighted means (and optionally uncertainties) grouped by detector type.
    """
    single_key = isinstance(value_keys, str)
    keys: list[str] = [value_keys] if single_key else list(value_keys)
    propagate_unc = unc_key is not None

    # accumulate lists per (detector type, key)
    type_data: dict[str, dict[str, dict[str, list]]] = {
        label: {k: {"vals": [], "w": [], "unc": []} for k in keys}
        for label in _DET_TYPE_MAP.values()
    }

    for period, period_dict in data.items():
        for run, run_dict in period_dict.items():
            for det, vals in run_dict.items():
                if (
                    usability_filter is not None
                    and vals.get("usability") != usability_filter
                ):
                    continue

                if weight_key not in vals:
                    logger.info(
                        "...found no weight key '%s' for %s in %s-%s, skip it",
                        weight_key,
                        det,
                        period,
                        run,
                    )
                    continue

                prefix = det[0].upper()
                det_type = _DET_TYPE_MAP.get(prefix)
                if det_type is None:
                    logger.warning(
                        "Unknown detector prefix '%s' for detector %s", prefix, det
                    )
                    continue

                w = vals[weight_key]
                sigma = vals.get(unc_key, 0.0) if propagate_unc else None
                if propagate_unc and not np.isfinite(sigma):
                    sigma = 0.0

                for k in keys:
                    if k not in vals:
                        logger.info(
                            "...found no key '%s' for %s in %s-%s, skip it",
                            k,
                            det,
                            period,
                            run,
                        )
                        continue
                    type_data[det_type][k]["vals"].append(vals[k])
                    type_data[det_type][k]["w"].append(w)
                    if propagate_unc:
                        type_data[det_type][k]["unc"].append(sigma)

    def _compute_for_key(det_type: str, key: str) -> float | dict | None:
        """Return weighted mean (and optionally uncertainty) for *key* and *det_type*."""
        d = type_data[det_type][key]
        if not d["vals"]:
            return fill_missing

        vals_arr = np.array(d["vals"], dtype=float)
        w_arr = np.array(d["w"], dtype=float)

        # drop non-finite value/weight pairs
        mask = np.isfinite(vals_arr) & np.isfinite(w_arr)
        vals_arr = vals_arr[mask]
        w_arr = w_arr[mask]

        total_w = np.sum(w_arr)
        if total_w == 0:
            logger.warning(
                "Total weight is zero for key '%s', det type %s", key, det_type
            )
            return fill_missing

        mean = weighted_mean(vals_arr, w_arr)

        if not propagate_unc:
            return mean

        # --- uncertainty propagation ---
        s_arr = np.array(d["unc"], dtype=float)[mask]
        # replace any remaining non-finite sigma with 0
        s_arr = np.where(np.isfinite(s_arr), s_arr, 0.0)

        unc_total = compute_weighted_uncertainty(w_arr, vals_arr, mean, s_arr)
        return {"value": mean, "unc": unc_total}

    result = {}
    for det_type in _DET_TYPE_MAP.values():
        result[det_type] = {}
        if single_key:
            out = _compute_for_key(det_type, keys[0])
            if out is not None:
                result[det_type][keys[0]] = out
        else:
            det_result = {}
            for k in keys:
                out = _compute_for_key(det_type, k)
                if out is not None:
                    det_result[k] = out
            if det_result:
                result[det_type] = det_result

    return result


@functools.cache
def get_channelmap_cached(meta: LegendMetadata, start_key: str):
    """Retrieve the channelmap, caching the result to avoid slow repeated filesystem queries."""
    return meta.channelmap(on=start_key)


def get_expo_per_detector(meta: LegendMetadata, periods_dict: dict) -> dict:
    """Build a nested dict with exposure and usability per detector.

    Parameters
    ----------
    meta : LegendMetadata
        Metadata object.
    periods_dict : dict
        Nested dict like ``{period: [run, ...]}``.

    Returns
    -------
    nested_dict : dict
        Nested dict like ``{period: {run: {detector: {"usability", "expo"}}}}``.
    """
    nested_dict = {}

    for period, runs_list in periods_dict.items():
        nested_dict[period] = {}
        for run in runs_list:
            nested_dict[period][run] = {}

            start_key = meta.datasets.runinfo[period][run].phy.start_key
            chmap = get_channelmap_cached(meta, start_key)
            ges = list(chmap.group("system")["geds"].map("name").keys())

            for ge in tqdm(ges, desc=f"{period}-{run} (expo)"):
                nested_dict[period][run][ge] = {}

                usability = chmap[ge]["analysis"]["usability"]
                nested_dict[period][run][ge]["usability"] = usability

                if usability != "on":
                    continue

                exp_kg_yr = get_exp_kg_yr(meta, period, run, ge)
                nested_dict[period][run][ge]["expo"] = exp_kg_yr

    return nested_dict


def get_eres_per_detector(
    meta: LegendMetadata, config: dict, periods_dict: dict
) -> dict:
    """Build a nested dict with energy resolution parameters and exposure per detector.

    Parameters
    ----------
    meta : LegendMetadata
        Metadata object.
    config : dict
        Configuration dict with file paths (must contain ``"par_pht"`` key).
    periods_dict : dict
        Nested dict like ``{period: [run, ...]}``.

    Returns
    -------
    nested_dict : dict
        Nested dict like ``{period: {run: {detector: {"usability", "expo", "a", "b", "a_unc", "b_unc", "ab_corr"}}}}``.
    """
    nested_dict = get_expo_per_detector(meta, periods_dict)

    for period, period_list in periods_dict.items():
        for run in period_list:
            start_key = meta.datasets.runinfo[period][run].phy.start_key
            chmap = get_channelmap_cached(meta, start_key)

            if (
                period == "p09" and run == "r004"
            ):  # Hardcoded cause p09-r004 calibration is missing from data
                timestamp_cal = meta.datasets.runinfo["p09"]["r005"].cal.start_key
                file_name = f"l200-p09-r005-cal-{timestamp_cal}-par_pht.json"
                pars = Props.read_from(f"{config['par_pht']}/cal/p09/r005/{file_name}")

            else:
                timestamp_cal = meta.datasets.runinfo[period][run].cal.start_key
                file_name = f"l200-{period}-{run}-cal-{timestamp_cal}-par_pht.json"
                pars = Props.read_from(
                    f"{config['par_pht']}/cal/{period}/{run}/{file_name}"
                )

            for ge, ge_data in tqdm(
                nested_dict[period][run].items(), desc=f"{period}-{run} (eres)"
            ):
                if ge_data.get("usability") != "on":
                    continue

                # get a and b
                a, b, a_unc, b_unc, cov = get_eres(chmap, ge, pars)
                ge_data["a"] = a
                ge_data["b"] = b
                ge_data["a_unc"] = a_unc
                ge_data["b_unc"] = b_unc
                ge_data["ab_corr"] = cov

    return nested_dict


def get_exp_kg_yr(meta: LegendMetadata, period: str, run: str, ge: str) -> float:
    """Return the exposure in kg·yr for a single detector in a single run.

    Parameters
    ----------
    meta : LegendMetadata
        Metadata object.
    period : str
        Data-taking period identifier (e.g. ``"p03"``).
    run : str
        Run identifier (e.g. ``"r000"``).
    ge : str
        Germanium detector name.

    Returns
    -------
    float
        Exposure in kg·yr.
    """
    run_livetime_in_s = meta.datasets.runinfo[period][run]["phy"]["livetime_in_s"]
    ge_mass_in_kg = (
        meta.hardware.detectors.germanium.diodes[ge].production.mass_in_g / 1000
    )

    return run_livetime_in_s * ge_mass_in_kg / 60 / 60 / 24 / 365.25


def get_eres(
    chmap: object, ge: str, pars: dict
) -> tuple[float, float, float, float, float]:
    """Extract energy resolution parameters ``(a, b)`` from calibration file.

    Parameters
    ----------
    chmap : object
        Channel map object (from :meth:`LegendMetadata.channelmap`).
    ge : str
        Germanium detector name.
    pars : dict
        Calibration parameters dict as loaded by :func:`dbetto.Props.read_from`.

    Returns
    -------
    a : float
        Constant term of the FWHM parametrisation.
    b : float
        Linear term of the FWHM parametrisation.
    cov : float
        Covariance between a and b.
    """
    channel = f"ch{chmap[ge].daq.rawid}"

    parameters = pars[channel]["results"]["partition_ecal"]["cuspEmax_ctc_cal"][
        "eres_linear"
    ]["parameters"]
    uncertainties = pars[channel]["results"]["partition_ecal"]["cuspEmax_ctc_cal"][
        "eres_linear"
    ]["uncertainties"]
    cov = pars[channel]["results"]["partition_ecal"]["cuspEmax_ctc_cal"]["eres_linear"][
        "cov"
    ][0][1]

    return parameters["a"], parameters["b"], uncertainties["a"], uncertainties["b"], cov

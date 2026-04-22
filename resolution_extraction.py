from __future__ import annotations

import logging

import numpy as np
from dbetto import Props
from legendmeta import LegendMetadata
from tqdm import tqdm

logger = logging.getLogger(__name__)


def weighted_resolution_from_nested_dict(data: dict, E: float) -> float:
    """
    Compute exposure-weighted FWHM resolution sqrt(a + b*E) from a nested dict structure.

    Parameters
    ----------
    config : dict
        Configuration file.
    data : dict
        Nested dict like {period: {run: {detector: {"expo", "a", "b", "a_unc", "b_unc"}}}}.
    E : float
        Energy at which to compute the resolution.

    Returns
    -------
    f_weighted : float
        Weighted mean FWHM at energy E.
    """
    a_list, b_list, w_list = [], [], []
    a_unc_list, b_unc_list = [], []

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

                a, b, w = vals["a"], vals["b"], vals["expo"]

                a_list.append(a)
                b_list.append(b)
                w_list.append(w)
                a_unc_list.append(vals.get("a_unc", 0.0))
                b_unc_list.append(vals.get("b_unc", 0.0))

    # arrays
    a = np.array(a_list)
    b = np.array(b_list)
    w = np.array(w_list)
    # TODO: uncertainty propagation
    # sa = np.array(a_unc_list)
    # sb = np.array(b_unc_list)
    # cab = np.zeros(len(a))  # null cov

    # per-detector FWHM
    f_i = np.sqrt(a + b * E)

    # TODO: propagate uncertainties
    # df_da = 1.0 / (2.0 * f_i)
    # df_db = E / (2.0 * f_i)
    # sigma_f_i = np.sqrt((df_da * sa) ** 2 + (df_db * sb) ** 2 + 2 * df_da * df_db * cab)

    # weighted average
    total_expo = np.sum(w)
    f_weighted = np.sum(w * f_i) / total_expo

    # TODO: propagate uncertainties
    # propagate measurement uncertainty to weighted mean
    # unc_meas = np.sqrt(np.sum((w * sigma_f_i) ** 2)) / total_expo

    # scatter uncertainty (not needed anymore)
    # var_w = np.sum(w * (f_i - f_weighted) ** 2) / total_expo
    # N_eff = total_expo**2 / np.sum(w**2)
    # unc_scatter = np.sqrt(var_w / N_eff)

    # total uncertainty
    # unc_total = np.sqrt(unc_meas**2 + unc_scatter**2)

    return f_weighted  # noqa: RET504


def weighted_resolution_per_detector(data: dict, E: float) -> dict:
    """Compute exposure-weighted FWHM resolution sqrt(a + b*E) per detector.

    Averaging over all period-run combinations.

    Parameters
    ----------
    data : dict
        Nested dict like {period: {run: {detector: {"expo", "a", "b", "a_unc", "b_unc"}}}}.
    E : float
        Energy at which to compute the resolution.

    Returns
    -------
    result : dict
        Dict like {detector: f_weighted}.
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
                    det_data[det] = {"a": [], "b": [], "w": []}

                det_data[det]["a"].append(vals["a"])
                det_data[det]["b"].append(vals["b"])
                det_data[det]["w"].append(vals["expo"])
                # det_data[det]["sa"].append(vals.get("a_unc", 0.0))
                # det_data[det]["sb"].append(vals.get("b_unc", 0.0))

    result = {}

    for det, d in det_data.items():
        a = np.array(d["a"])
        b = np.array(d["b"])
        w = np.array(d["w"])
        # sa = np.array(d["sa"])
        # sb = np.array(d["sb"])

        # per-run FWHM
        f_i = np.sqrt(a + b * E)

        # propagate uncertainties
        # df_da = 1.0 / (2.0 * f_i)
        # df_db = E / (2.0 * f_i)
        # sigma_f_i = np.sqrt((df_da * sa) ** 2 + (df_db * sb) ** 2)

        # weighted average
        total_expo = np.sum(w)
        f_weighted = np.sum(w * f_i) / total_expo

        # propagate measurement uncertainty to weighted mean
        # unc_meas = np.sqrt(np.sum((w * sigma_f_i) ** 2)) / total_expo

        # scatter uncertainty
        # var_w = np.sum(w * (f_i - f_weighted) ** 2) / total_expo
        # N_eff = total_expo ** 2 / np.sum(w ** 2)
        # unc_scatter = np.sqrt(var_w / N_eff)

        # total uncertainty
        # unc_total = np.sqrt(unc_meas ** 2 + unc_scatter ** 2)

        result[det] = f_weighted

    return result


def get_eres_per_detector(meta: LegendMetadata, config: dict, periods_dict: dict):
    """
    Compute exposure per detector from a nested dict structure.

    Parameters
    ----------
    meta : LegendMetadata
        Metadata object
    config : dict
        Configuration file with paths.
    periods_dict : dict
        Nested dict like {period: {"runs"}}

    Returns
    -------
    nested_dict : dict
        Nested dict like {period: {run: {detector: {"usability", "expo", "a", "b"}}}}
    """
    nested_dict = {}

    for period, period_list in periods_dict.items():
        nested_dict[period] = {}
        for run in period_list:
            nested_dict[period][run] = {}

            start_key = meta.datasets.runinfo[period][run].phy.start_key
            chmap = meta.channelmap(on=start_key)
            ges = list(chmap.group("system")["geds"].map("name").keys())

            timestamp_cal = meta.datasets.runinfo[period][run].cal.start_key
            file_name = f"l200-{period}-{run}-cal-{timestamp_cal}-par_pht.json"
            pars = Props.read_from(
                f"{config['par_pht']}/cal/{period}/{run}/{file_name}"
            )

            for ge in tqdm(ges, desc=f"{period}-{run}"):
                nested_dict[period][run][ge] = {}

                usability = chmap[ge]["analysis"]["usability"]
                nested_dict[period][run][ge]["usability"] = usability

                if usability != "on":
                    continue

                exp_kg_yr = get_exp_kg_yr(meta, period, run, ge)
                nested_dict[period][run][ge]["expo"] = exp_kg_yr

                # get a and b
                a, b = get_eres(chmap, ge, pars)
                nested_dict[period][run][ge]["a"] = a
                nested_dict[period][run][ge]["b"] = b

    return nested_dict


def get_exp_kg_yr(meta, period, run, ge):
    run_livetime_in_s = meta.datasets.runinfo[period][run]["phy"]["livetime_in_s"]
    ge_mass_in_kg = (
        meta.hardware.detectors.germanium.diodes[ge].production.mass_in_g / 1000
    )

    return run_livetime_in_s * ge_mass_in_kg / 60 / 60 / 24 / 365.25


def get_eres(chmap, ge, pars):
    channel = f"ch{chmap[ge].daq.rawid}"

    parameters = pars[channel]["results"]["partition_ecal"]["cuspEmax_ctc_cal"][
        "eres_linear"
    ]["parameters"]

    return parameters["a"], parameters["b"]

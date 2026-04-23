from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from dbetto import Props
from legendmeta import LegendMetadata
from tqdm import tqdm

logger = logging.getLogger(__name__)


def weighted_resolution_from_nested_dict(data: dict, E: float) -> float:
    """Compute exposure-weighted FWHM resolution sqrt(a + b*E) from a nested dict structure.

    Parameters
    ----------
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

                a_val, b_val, w_val = vals["a"], vals["b"], vals["expo"]

                a_list.append(a_val)
                b_list.append(b_val)
                w_list.append(w_val)
                a_unc_list.append(vals.get("a_unc", 0.0))
                b_unc_list.append(vals.get("b_unc", 0.0))

    # arrays
    a_arr = np.array(a_list)
    b_arr = np.array(b_list)
    w_arr = np.array(w_list)

    a_unc_arr = np.array(a_unc_list)
    b_unc_arr = np.array(b_unc_list)
    cab = np.zeros(len(a_arr))  # null cov

    # per-detector FWHM
    f_i = np.sqrt(a_arr + b_arr * E)

    # propagate uncertainties
    df_da = 1.0 / (2.0 * f_i)
    df_db = E / (2.0 * f_i)
    sigma_f_i = np.sqrt(
        (df_da * a_unc_arr) ** 2 + (df_db * b_unc_arr) ** 2 + 2 * df_da * df_db * cab
    )

    # weighted average
    total_expo = np.sum(w_arr)
    f_weighted = np.sum(w_arr * f_i) / total_expo

    # propagate measurement uncertainty to weighted mean
    unc_meas = np.sqrt(np.sum((w_arr * sigma_f_i) ** 2)) / total_expo

    # scatter uncertainty (not needed anymore)
    var_w = np.sum(w_arr * (f_i - f_weighted) ** 2) / total_expo
    N_eff = total_expo**2 / np.sum(w_arr**2)
    unc_scatter = np.sqrt(var_w / N_eff)

    # total uncertainty
    unc_total = np.sqrt(unc_meas**2 + unc_scatter**2)

    return f_weighted, unc_total


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
                    det_data[det] = {"a": [], "b": [], "w": []}

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
        cab = np.array(d["cab"])

        # per-run sigma
        fwhm_arr = compute_fwhm(a_arr, b_arr, E)

        # propagate uncertainties
        df_da = 1.0 / (2.0 * fwhm_arr)
        df_db = E / (2.0 * fwhm_arr)
        sigma_f_arr = np.sqrt(
            (df_da * sa_arr) ** 2 + (df_db * sb_arr) ** 2 + 2 * df_da * df_db * cab
        )

        # weighted average
        total_expo = np.sum(w_arr)
        fwhm_weighted = np.sum(w_arr * fwhm_arr) / total_expo

        # propagate measurement uncertainty to weighted mean
        unc_meas = np.sqrt(np.sum((w_arr * sigma_f_arr) ** 2)) / total_expo

        # scatter uncertainty
        var_w = np.sum(w_arr * (fwhm_arr - fwhm_weighted) ** 2) / total_expo
        N_eff = total_expo**2 / np.sum(w_arr**2)
        unc_scatter = np.sqrt(var_w / N_eff)

        # total uncertainty
        unc_total = np.sqrt(unc_meas**2 + unc_scatter**2)

        result[det] = {"fwhm": fwhm_weighted, "unc": unc_total, "expo": total_expo}

    return result


# detector-name prefix → human-readable type label
_DET_TYPE_MAP = {"B": "BEGe", "C": "COAX", "V": "ICPC", "P": "PPC"}


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

        * If a **single string** is passed, the return dict has the shape
          ``{det_type: weighted_mean}`` (or ``{det_type: {"value": ..., "unc": ...}}``
          when *unc_key* is set).
        * If a **list of strings** is passed, the return dict has the shape
          ``{det_type: {key: weighted_mean}}`` (or
          ``{det_type: {key: {"value": ..., "unc": ...}}}`` when *unc_key* is set),
          computed in a single traversal.
    weight_key : str, optional
        Key used as the exposure weight. Defaults to ``"expo"``.
    unc_key : str or None, optional
        If provided, the key in the innermost dict that holds the pre-computed
        uncertainty on each entry.  When set, two uncertainty components are
        propagated and added in quadrature:

        * **Measurement**: ``unc_meas = sqrt(sum((w_i * sigma_i)^2)) / sum(w_i)``
          — propagates the per-entry uncertainty through the weighted average.
        * **Scatter**: ``unc_scatter = sqrt(Var_w / N_eff)`` where
          ``Var_w = sum(w_i * (v_i - mean)^2) / sum(w_i)`` and
          ``N_eff = (sum w_i)^2 / sum(w_i^2)`` — captures the spread among
          entries of the same detector type.

        The total uncertainty is ``unc_total = sqrt(unc_meas^2 + unc_scatter^2)``.
        Entries whose uncertainty is missing or non-finite are treated as 0.
        Defaults to ``None`` (no uncertainty propagation).
    usability_filter : str or None, optional
        If not ``None``, only detector entries whose ``"usability"`` field
        equals this value are included. Pass ``None`` to skip the check entirely.
        Defaults to ``"on"``.
    fill_missing : float or None, optional
        Value to use for detector types with no valid data (e.g. ``np.nan``).
        If ``None`` (default) those types are omitted from the result.

    Returns
    -------
    result : dict
        Without *unc_key*: ``{det_type: weighted_mean}`` or
        ``{det_type: {key: weighted_mean}}``.

        With *unc_key*: ``{det_type: {"value": weighted_mean, "unc": unc_total}}`` or
        ``{det_type: {key: {"value": weighted_mean, "unc": unc_total}}}``.
    """
    single_key = isinstance(value_keys, str)
    keys: list[str] = [value_keys] if single_key else list(value_keys)
    propagate_unc = unc_key is not None

    # accumulate lists per (detector type, key)
    # Each entry stores vals, weights, and optionally per-entry uncertainties.
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

        mean = float(np.sum(w_arr * vals_arr) / total_w)

        if not propagate_unc:
            return mean

        # --- uncertainty propagation ---
        s_arr = np.array(d["unc"], dtype=float)[mask]
        # replace any remaining non-finite sigma with 0
        s_arr = np.where(np.isfinite(s_arr), s_arr, 0.0)

        # measurement component: quadrature propagation through weighted average
        unc_meas = float(np.sqrt(np.sum((w_arr * s_arr) ** 2)) / total_w)

        # scatter component: weighted variance across entries
        var_w = float(np.sum(w_arr * (vals_arr - mean) ** 2) / total_w)
        n_eff = float(total_w**2 / np.sum(w_arr**2))
        unc_scatter = float(np.sqrt(var_w / n_eff)) if n_eff > 0 else 0.0

        unc_total = float(np.sqrt(unc_meas**2 + unc_scatter**2))
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


def plot_resolution_per_det_type(
    results_dict: dict[str, dict[str, float]],
    ene_values: list[float],
    fig_name: str | None = None,
) -> None:
    """Plot FWHM as a function of energy for each detector type.

    Parameters
    ----------
    results_dict : dict
        Nested dict with structure ``{det_type: {ene: fwhm}}``.
    ene_values : list of float
        List of energies at which to plot.
    fig_name : str or None, optional
        Path of the figure to save. If ``None`` (default), the figure is not saved.
    """
    plt.figure(figsize=(10, 6))
    for det_type in results_dict:
        if "fwhm" in results_dict[det_type][ene_values[0]].keys():
            fwhms = [results_dict[det_type][ene]["fwhm"] for ene in ene_values]
        else:
            fwhms = [results_dict[det_type][ene] for ene in ene_values]

        plt.plot(ene_values, fwhms, label=det_type, marker="o", linestyle="--")

    plt.xlabel("Energy [keV]")
    plt.ylabel("FWHM [keV]")
    plt.legend()
    if fig_name is not None:
        plt.savefig(fig_name, dpi=400)
    plt.show()


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
        Nested dict like ``{period: {run: {detector: {"usability", "expo", "a", "b"}}}}``.
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
                a, b, a_unc, b_unc = get_eres(chmap, ge, pars)
                nested_dict[period][run][ge]["a"] = a
                nested_dict[period][run][ge]["b"] = b
                nested_dict[period][run][ge]["a_unc"] = a_unc
                nested_dict[period][run][ge]["b_unc"] = b_unc

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


def get_eres(chmap: object, ge: str, pars: dict) -> tuple[float, float]:
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
    """
    channel = f"ch{chmap[ge].daq.rawid}"

    parameters = pars[channel]["results"]["partition_ecal"]["cuspEmax_ctc_cal"][
        "eres_linear"
    ]["parameters"]
    uncertainties = pars[channel]["results"]["partition_ecal"]["cuspEmax_ctc_cal"][
        "eres_linear"
    ]["uncertainties"]

    return parameters["a"], parameters["b"], uncertainties["a"], uncertainties["b"]

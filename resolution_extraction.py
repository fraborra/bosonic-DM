from __future__ import annotations

import functools
import logging

import matplotlib.pyplot as plt
import numpy as np
from dbetto import Props
from helper_lib import compute_weighted_uncertainty, weighted_mean
from legendmeta import LegendMetadata
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

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


# detector-name prefix → human-readable type label
_DET_TYPE_MAP = {"B": "BEGe", "C": "COAX", "V": "ICPC", "P": "PPC"}
_DET_TYPE_COLOR = {
    "BEGe": "tab:blue",
    "COAX": "tab:orange",
    "ICPC": "tab:green",
    "PPC": "tab:red",
}


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
            uncs = [results_dict[det_type][ene]["unc"] for ene in ene_values]
            plt.errorbar(
                ene_values, fwhms, yerr=uncs, label=det_type, marker=".", linestyle="--"
            )
        else:
            fwhms = [results_dict[det_type][ene] for ene in ene_values]
            plt.plot(ene_values, fwhms, label=det_type, marker=".", linestyle="--")

    plt.xlabel("Energy [keV]")
    plt.ylabel("FWHM [keV]")
    plt.legend()
    if fig_name is not None:
        plt.savefig(fig_name, dpi=400)
    plt.show()


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

    for period, period_list in periods_dict.items():
        nested_dict[period] = {}
        for run in period_list:
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


def plot_fwhm_vs_period_run(
    data: dict,
    energy: float = 1000.0,
    detectors: str | list[str] | None = None,
    pdf_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot FWHM vs period-run for one, several, or all detectors.

    Parameters
    ----------
    data : dict
        Nested dict ``{period: {run: {detector: {"a", "b", "usability", ...}}}}``.
    energy : float, optional
        Energy (keV) at which to evaluate ``sqrt(a + b*E)``.  Default is 2039 keV.
    detectors : str, list of str, or None
        * ``str``  - plot a single detector.
        * ``list`` - plot each detector in the list.
        * ``None`` - plot **all** detectors that appear at least once with
          ``usability == "on"``.
    pdf_path : str or None, optional
        If given, every figure is saved into a single multi-page PDF at this path.
        When ``None`` (default), figures are not saved.
    show : bool, optional
        Whether to call ``plt.show()`` after each figure.  Set to ``False`` when
        running non-interactively and only saving to PDF.
    """
    # ── 1. build a sorted list of (period, run) labels ──────────────────
    period_runs: list[tuple[str, str]] = []
    for period in sorted(data.keys()):
        for run in sorted(data[period].keys()):
            period_runs.append((period, run))

    x_labels = [f"{p}-{r}" for p, r in period_runs]

    # ── 2. determine which detectors to plot ────────────────────────────
    if detectors is None:
        det_set: set[str] = set()
        for _period, pdict in data.items():
            for _run, rdict in pdict.items():
                for det, vals in rdict.items():
                    if vals.get("usability") == "on" and "a" in vals and "b" in vals:
                        det_set.add(det)
        det_list = sorted(det_set)
    elif isinstance(detectors, str):
        det_list = [detectors]
    else:
        det_list = list(detectors)

    if not det_list:
        logger.warning("No detectors to plot.")
        return

    # ── 3. plot ─────────────────────────────────────────────────────────
    pdf = PdfPages(pdf_path) if pdf_path else None

    try:
        for det in det_list:
            fwhm_vals: list[float | None] = []
            unc_vals: list[float | None] = []

            for period, run in period_runs:
                vals = data.get(period, {}).get(run, {}).get(det)
                if (
                    vals is None
                    or vals.get("usability") != "on"
                    or "a" not in vals
                    or "b" not in vals
                ):
                    fwhm_vals.append(None)
                    unc_vals.append(None)
                else:
                    a_val = vals["a"]
                    b_val = vals["b"]
                    fwhm = compute_fwhm(a_val, b_val, energy)
                    fwhm_vals.append(fwhm)

                    a_unc = vals.get("a_unc", 0.0)
                    b_unc = vals.get("b_unc", 0.0)
                    ab_corr = vals.get("ab_corr", 0.0)
                    unc = propagate_resolution_uncertainty(
                        a_val, b_val, a_unc, b_unc, ab_corr, energy
                    )
                    unc_vals.append(float(unc))

            # separate valid / missing points
            x_pos = np.arange(len(x_labels))
            mask = np.array([v is not None for v in fwhm_vals])

            if not mask.any():
                logger.info("Detector %s has no valid data - skipping.", det)
                continue

            y_plot = np.array([v if v is not None else 0.0 for v in fwhm_vals])
            e_plot = np.array([v if v is not None else 0.0 for v in unc_vals])

            det_type = _DET_TYPE_MAP.get(det[0].upper(), "unknown")

            color = _DET_TYPE_COLOR.get(det_type, "tab:gray")

            fig, ax = plt.subplots(figsize=(max(8, len(x_labels) * 0.45), 5))
            ax.errorbar(
                x_pos[mask],
                y_plot[mask],
                yerr=e_plot[mask],
                fmt="o",
                capsize=3,
                color=color,
                label=f"{det} ({det_type})",
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=60, ha="right", fontsize=8)
            ax.set_xlabel("Period - Run")
            ax.set_ylabel(f"FWHM @{energy} keV [keV]")
            ax.set_title(f"FWHM vs Period-Run — {det}")
            ax.legend()
            fig.tight_layout()

            if pdf is not None:
                pdf.savefig(fig)
            if show:
                plt.show()
            else:
                plt.close(fig)
    finally:
        if pdf is not None:
            pdf.close()
            logger.info("Saved multi-page PDF to %s", pdf_path)


def plot_fwhm_vs_period_run_overlay(
    data: dict,
    detectors: list[str],
    energy: float = 1000.0,
    fig_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot FWHM vs period-run for several detectors on the same figure.

    Parameters
    ----------
    data : dict
        Nested dict ``{period: {run: {detector: {"a", "b", "usability", ...}}}}``.
    detectors : list of str
        Detectors to overlay on the same axes.
    energy : float, optional
        Energy (keV) at which to evaluate ``sqrt(a + b*E)``.  Default is 1000 keV.
    fig_path : str or None, optional
        If given, the figure is saved to this path.  When ``None`` (default),
        the figure is not saved.
    show : bool, optional
        Whether to call ``plt.show()``.  Set to ``False`` when running
        non-interactively and only saving to file.
    """
    # ── 1. build a sorted list of (period, run) labels ──────────────────
    period_runs: list[tuple[str, str]] = []
    for period in sorted(data.keys()):
        for run in sorted(data[period].keys()):
            period_runs.append((period, run))

    x_labels = [f"{p}-{r}" for p, r in period_runs]
    x_pos = np.arange(len(x_labels))

    if not detectors:
        logger.warning("No detectors to plot.")
        return

    # ── 2. unique color per detector ────────────────────────────────────
    cmap = plt.get_cmap("tab10") if len(detectors) <= 10 else plt.get_cmap("tab20")
    _markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]

    fig, ax = plt.subplots(figsize=(max(8, len(x_labels) * 0.45), 5))

    for idx, det in enumerate(detectors):
        fwhm_vals: list[float | None] = []
        unc_vals: list[float | None] = []

        for period, run in period_runs:
            vals = data.get(period, {}).get(run, {}).get(det)
            if (
                vals is None
                or vals.get("usability") != "on"
                or "a" not in vals
                or "b" not in vals
            ):
                fwhm_vals.append(None)
                unc_vals.append(None)
            else:
                a_val = vals["a"]
                b_val = vals["b"]
                fwhm = compute_fwhm(a_val, b_val, energy)
                fwhm_vals.append(fwhm)

                a_unc = vals.get("a_unc", 0.0)
                b_unc = vals.get("b_unc", 0.0)
                ab_corr = vals.get("ab_corr", 0.0)
                unc = propagate_resolution_uncertainty(
                    a_val, b_val, a_unc, b_unc, ab_corr, energy
                )
                unc_vals.append(float(unc))

        mask = np.array([v is not None for v in fwhm_vals])
        if not mask.any():
            logger.info("Detector %s has no valid data - skipping.", det)
            continue

        y_plot = np.array([v if v is not None else 0.0 for v in fwhm_vals])
        e_plot = np.array([v if v is not None else 0.0 for v in unc_vals])

        color = cmap(idx % cmap.N)
        marker = _markers[idx % len(_markers)]

        ax.errorbar(
            x_pos[mask],
            y_plot[mask],
            yerr=e_plot[mask],
            fmt=marker,
            capsize=3,
            color=color,
            label=det,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("Period - Run")
    ax.set_ylabel(f"FWHM @{energy} keV [keV]")
    ax.set_title("FWHM vs Period-Run")
    ncol = int(np.ceil(len(detectors) / 15))
    # Place the legend to the right of the plot area
    ax.legend(
        bbox_to_anchor=(1.02, 1.0), loc="upper left", ncol=ncol, borderaxespad=0.0
    )
    # Use tight layout to automaically make room for the legend on the right
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path, dpi=400)
    if show:
        plt.show()
    else:
        plt.close(fig)

# Copyright (C) 2025 Francesco Borra
#

"""Resolution plotting functions for bosonic-DM analysis."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from bosonic_dm.plotting.utils import _DET_TYPE_COLOR, _DET_TYPE_MAP
from bosonic_dm.resolution import compute_fwhm, propagate_resolution_uncertainty

logger = logging.getLogger(__name__)


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

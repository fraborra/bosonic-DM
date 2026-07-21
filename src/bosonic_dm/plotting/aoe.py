# Copyright (C) 2025 Francesco Borra
#

"""AoE plotting functions for bosonic-DM analysis."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from tqdm.notebook import tqdm

from bosonic_dm.io import get_rawid_lists
from bosonic_dm.plotting.utils import _DET_TYPE_COLOR, _DET_TYPE_MAP

logger = logging.getLogger(__name__)


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
        else:
            aoe_all_clean = np.array([])

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

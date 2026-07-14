# Copyright (C) 2025 Francesco Borra
#

"""Spectra plotting functions for bosonic-DM analysis."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from legendmeta import LegendMetadata
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.notebook import tqdm

from bosonic_dm.cuts import compute_group_exposure
from bosonic_dm.plotting.utils import _DET_TYPE_COLOR, _DET_TYPE_MAP
from bosonic_dm.stats import bayesian_efficiency

logger = logging.getLogger(__name__)

_FEP_COLORS = {
    "e$^-$ + $\\gamma$": "red",
    "$\\gamma$": "green",
    "e$^-$": "purple",
}


def plot_detector_spectra(
    df_all: pl.DataFrame,
    df_no_qc: pl.DataFrame,
    eres_dict: Mapping,
    energy_range: Sequence[float],
    bins: int,
    pdf_filename: str | Path,
    yscale: str = "log",
) -> None:
    """Plot the energy spectra for each detector, pre and post QC, saving to a single PDF.

    Parameters
    ----------
    df_all
        DataFrame containing all events (post-QC).
    df_no_qc
        DataFrame containing events before QC (pre-QC).
    eres_dict
        Exposure/resolution dictionary loaded from eres_dict.yaml.
    energy_range
        A sequence/tuple containing (emin, emax) for the energy range in keV.
    bins
        The number of bins to use in the histogram.
    yscale
        The scale to use for the y-axis. Can be "linear" or "log".
    pdf_filename
        The output path for the PDF file.
    """
    metadata_path = "/global/homes/b/borrfran/workspace/l200/legend-metadata"
    meta = LegendMetadata(metadata_path)

    emin, emax = energy_range
    bin_width = (emax - emin) / bins

    # Get a sorted list of unique detectors from the pre-QC dataset
    all_detectors = df_all["detector_name"].unique().to_list()
    # Skip any "unknown" detector names
    detectors = sorted([det for det in all_detectors if det != "unknown"])

    with PdfPages(pdf_filename) as pdf:
        for det_name in tqdm(detectors):
            # Determine detector type and color
            det_type = _DET_TYPE_MAP.get(det_name[0].upper(), "unknown")
            color = _DET_TYPE_COLOR.get(det_type, "tab:gray")

            # Filter data for this detector
            df_det_all = df_all.filter(pl.col("detector_name") == det_name)
            df_det_no_qc = df_no_qc.filter(pl.col("detector_name") == det_name)

            energy_all = df_det_all["energy"].to_numpy()
            energy_no_qc = df_det_no_qc["energy"].to_numpy()

            # Compute exposure
            expo = compute_group_exposure(eres_dict, {det_name: "all"})

            fig, ax = plt.subplots(figsize=(10, 6))

            if expo > 0:
                weights_all = np.ones_like(energy_all) / (expo * bin_width)
                weights_no_qc = np.ones_like(energy_no_qc) / (expo * bin_width)
                y_label = "Counts / (keV · kg · yr)"
                title_suffix = f"Exposure: {expo:.3g} kg·yr"
            else:
                weights_all = None
                weights_no_qc = None
                y_label = "Counts / keV"
                title_suffix = "Exposure: N/A"

            mass = (
                meta.hardware.detectors.germanium.diodes[det_name].production.mass_in_g
                / 1000
            )
            title_suffix_mass = f"Mass: {mass:.3g} kg"
            # Plot pre-QC (no QC) - solid line
            ax.hist(
                energy_no_qc,
                bins=bins,
                range=(emin, emax),
                histtype="step",
                linewidth=1.5,
                color=color,
                linestyle="-",
                weights=weights_no_qc,
                label=f"{det_name} (no QC)",
            )

            # Plot post-QC (QC) - dashed line
            ax.hist(
                energy_all,
                bins=bins,
                range=(emin, emax),
                histtype="step",
                linewidth=1.5,
                color=color,
                linestyle="--",
                weights=weights_all,
                label=f"{det_name} (QC)",
            )

            ax.set_xlabel("Energy [keV]", fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_title(
                f"Energy Spectrum for {det_name} — {title_suffix} - {title_suffix_mass}",
                fontsize=13,
            )
            ax.set_yscale(yscale)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(frameon=True, facecolor="white", edgecolor="none")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


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
    x_range
        Optional ``(low, high)`` tuple for the x-axis range.  If
        *None*, the range is determined from the data.
    save_dir
        Directory where figures are saved.  Figures are named
        ``lar_survival_fraction_{ene}keV.png``.
    """
    from bosonic_dm.dark_compton_generators import calculate_energies  # noqa: PLC0415

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

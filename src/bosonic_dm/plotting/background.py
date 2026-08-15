# Copyright (C) 2025 Francesco Borra
#

"""Diagnostic check plots for the background Parquet dataset."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def plot_background_spectrum(
    dataset_path: Path,
    energy_range_keV: tuple[int, int],
    bin_width_keV: int,
    comparison_cut_profile: str,
    output_dir: Path,
    *,
    exposure_kg_yr: float | None = None,
    yscale: str = "log",
    show_plot: bool = False,
) -> Path:
    """Plot an analysis-vs-comparison spectrum with a ratio panel.

    Parameters
    ----------
    dataset_path
        Root of the Hive-partitioned background Parquet dataset.
    energy_range_keV
        ``(emin, emax)`` energy window.
    bin_width_keV
        Histogram bin width in keV.
    comparison_cut_profile
        Label for the comparison selection (e.g. ``"without-bb-like"``).
    output_dir
        Directory where the figure PNG is saved.
    exposure_kg_yr
        Optional total exposure in kg·yr.  When provided the y-axis
        is normalised to counts / (keV · kg · yr); otherwise raw
        counts / keV are shown.
    yscale
        Scale for the spectrum y-axis (``"log"`` or ``"linear"``).
    show_plot
        If ``True``, display the figure interactively (e.g. in a
        notebook) instead of closing it after saving.

    Returns
    -------
    Path
        Absolute path of the saved PNG file.
    """
    emin, emax = energy_range_keV
    n_bins = max(1, int((emax - emin) / bin_width_keV))
    bin_edges = np.linspace(emin, emax, n_bins + 1)

    lf = pl.scan_parquet(str(dataset_path), hive_partitioning=True)
    df = (
        lf.filter(
            (pl.col("energy") >= emin) & (pl.col("energy") < emax)
        )
        .select("energy", "passes_analysis", "passes_comparison")
        .collect()
    )

    energy = df["energy"].to_numpy()
    mask_analysis = df["passes_analysis"].to_numpy()
    mask_comparison = df["passes_comparison"].to_numpy()

    counts_analysis, _ = np.histogram(energy[mask_analysis], bins=bin_edges)
    counts_comparison, _ = np.histogram(energy[mask_comparison], bins=bin_edges)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Normalisation
    if exposure_kg_yr is not None and exposure_kg_yr > 0:
        norm = exposure_kg_yr * bin_width_keV
        ylabel = "Counts / (keV · kg · yr)"
    else:
        norm = bin_width_keV
        ylabel = "Counts / keV"

    vals_analysis = counts_analysis / norm
    vals_comparison = counts_comparison / norm

    # --- Figure ---------------------------------------------------------------
    fig, (ax_spec, ax_ratio) = plt.subplots(
        2, 1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    # Top panel: overlay spectra
    ax_spec.stairs(
        vals_analysis,
        bin_edges,
        linewidth=1.4,
        color="tab:blue",
        label="analysis (default + LAr)",
    )
    ax_spec.stairs(
        vals_comparison,
        bin_edges,
        linewidth=1.4,
        color="tab:orange",
        label=f"comparison ({comparison_cut_profile})",
    )
    ax_spec.set_ylabel(ylabel, fontsize=11)
    ax_spec.set_yscale(yscale)
    ax_spec.legend(fontsize=9, loc="upper right", frameon=True)
    # ax_spec.grid(True, linestyle=":", alpha=0.5)
    ax_spec.set_title(
        f"Background spectrum  [{emin}–{emax}] keV, {bin_width_keV} keV bins",
        fontsize=12,
    )

    # Bottom panel: ratio
    safe = counts_analysis > 0
    ratio = np.full_like(counts_analysis, np.nan, dtype=float)
    ratio_err = np.full_like(counts_analysis, np.nan, dtype=float)

    ratio[safe] = counts_comparison[safe] / counts_analysis[safe]
    # Poisson-propagated uncertainty: sigma(r) = r * sqrt(1/n_c + 1/n_a)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_err[safe] = ratio[safe] * np.sqrt(
            1.0 / counts_comparison[safe] + 1.0 / counts_analysis[safe]
        )
    # Guard against NaN from zero comparison counts
    ratio_err = np.nan_to_num(ratio_err, nan=0.0)

    ax_ratio.errorbar(
        bin_centres[safe],
        ratio[safe],
        yerr=ratio_err[safe],
        fmt=".",
        markersize=4,
        linewidth=0.8,
        color="tab:purple",
    )
    # Mark bins with zero analysis events
    empty = ~safe
    if np.any(empty):
        ax_ratio.plot(
            bin_centres[empty],
            np.zeros(int(np.sum(empty))),
            marker="x",
            linestyle="none",
            color="red",
            markersize=5,
            label="no analysis events",
        )
        ax_ratio.legend(fontsize=8, loc="upper right")

    ax_ratio.axhline(1.0, color="grey", linestyle="--", linewidth=0.7)
    ax_ratio.set_xlabel("Energy [keV]", fontsize=11)
    ax_ratio.set_ylabel("comparison / analysis", fontsize=10)
    ax_ratio.set_ylim(-0.1, 2.1)
    # ax_ratio.grid(True, linestyle=":", alpha=0.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"spectrum_{emin}_{emax}keV_{bin_width_keV}keV.png"
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    logger.info("Background spectrum saved to %s", filename)
    return filename


def plot_background_partition_summary(
    dataset_path: Path,
    output_dir: Path,
    *,
    exposure_by_partition: dict[tuple[str, str], float] | None = None,
    show_plot: bool = False,
) -> tuple[Path, list[str]]:
    """Bar chart of event counts per (period, run) partition.

    Parameters
    ----------
    dataset_path
        Root of the Hive-partitioned background Parquet dataset.
    output_dir
        Directory where the figure PNG is saved.
    exposure_by_partition
        Optional mapping of ``(period, run)`` to exposure in kg·yr.
        When provided, the bar values are normalised to
        counts / (kg · yr) so that partitions with different
        exposures can be compared directly.
    show_plot
        If ``True``, display the figure interactively instead of
        closing it after saving.

    Returns
    -------
    tuple[Path, list[str]]
        The saved PNG path and a list of warning strings for partitions
        with zero ``passes_analysis`` events.
    """
    lf = pl.scan_parquet(str(dataset_path), hive_partitioning=True)
    summary = (
        lf.group_by("period", "run")
        .agg(
            pl.len().alias("total"),
            pl.col("passes_analysis").sum().alias("n_analysis"),
            pl.col("passes_comparison").sum().alias("n_comparison"),
        )
        .sort("period", "run")
        .collect()
    )

    periods = summary["period"].to_list()
    runs = summary["run"].to_list()
    labels = [f"{p}/{r}" for p, r in zip(periods, runs)]
    total = summary["total"].to_numpy().astype(float)
    n_analysis = summary["n_analysis"].to_numpy().astype(float)
    n_comparison = summary["n_comparison"].to_numpy().astype(float)

    warnings: list[str] = []
    for i, label in enumerate(labels):
        if n_analysis[i] == 0:
            warnings.append(
                f"Partition {label} has zero events passing analysis cuts"
            )

    # Normalise by exposure when available
    if exposure_by_partition:
        for i, (p, r) in enumerate(zip(periods, runs)):
            expo = exposure_by_partition.get((p, r))
            if expo is not None and expo > 0:
                total[i] /= expo
                n_analysis[i] /= expo
                n_comparison[i] /= expo
        ylabel = "Counts / (kg · yr)"
        title = "Background partitions — event rate"
    else:
        ylabel = "Event count"
        title = "Background partitions — event counts"

    x = np.arange(len(labels))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5))
    ax.bar(x - bar_width, total, bar_width, label="total", color="tab:gray")
    ax.bar(x, n_analysis, bar_width, label="passes_analysis", color="tab:blue")
    ax.bar(
        x + bar_width,
        n_comparison,
        bar_width,
        label="passes_comparison",
        color="tab:orange",
    )

    # Flag empty partitions
    empty_mask = n_analysis == 0
    if np.any(empty_mask):
        ax.scatter(
            x[empty_mask],
            np.zeros(int(np.sum(empty_mask))),
            marker="x",
            color="red",
            s=80,
            zorder=5,
            label="empty analysis",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    # ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / "partition_summary.png"
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    logger.info("Background partition summary saved to %s", filename)
    return filename, warnings

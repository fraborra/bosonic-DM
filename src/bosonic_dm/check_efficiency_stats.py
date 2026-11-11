# Copyright (C) 2026 Francesco Borra
#

"""Rank detectors by the statistical uncertainty of their simulated efficiency.

The command reads the efficiency table written by the ``efficiencies`` stage
of the simulation pipeline. For every simulated energy it lists the detectors
with the largest relative and absolute statistical uncertainty, and how many
exceed a target relative uncertainty. Use it to decide whether more events
must be simulated::

    pixi run check-efficiency-stats --config FILE --interaction dark-compton
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from bosonic_dm.config import load_analysis_config
from bosonic_dm.constants import DET_TYPE_MAP
from bosonic_dm.efficiency import (
    combine_run_efficiencies,
    efficiency_table_path,
    load_efficiency_table,
)

logger = logging.getLogger(__name__)


def efficiency_stat_table(table: pl.DataFrame, selection: str = "all") -> pl.DataFrame:
    """Return detector-level efficiencies with their statistical uncertainty.

    The runs of a detector are combined as in the efficiency aggregation
    (``bosonic_dm.efficiency.combine_run_efficiencies``): exposure-weighted
    counts, then one Jeffreys posterior for the detector.

    Parameters
    ----------
    table
        Efficiency table from ``bosonic_dm.efficiency.load_efficiency_table``.
    selection
        Efficiency selection to summarize.

    Returns
    -------
    pl.DataFrame
        One row per simulated energy and detector with a valid result, sorted
        by both, with columns ``sim_e``, ``detector``, ``detector_type``,
        ``efficiency``, ``efficiency_stat_unc``, ``relative_unc``,
        ``n_primaries``, and ``n_events`` (both summed over the detector's
        valid runs).
    """
    runs = (
        table.filter(
            (pl.col("selection") == selection)
            & (pl.col("status") == "valid")
            & (pl.col("exposure_kg_yr") > 0)
        )
        .drop_nulls(["n_events", "n_primaries"])
        .sort("sim_e", "detector", "period", "run")
    )
    rows: list[dict[str, object]] = []
    for (sim_e, detector), detector_runs in runs.group_by(
        ["sim_e", "detector"], maintain_order=True
    ):
        combined = combine_run_efficiencies(
            detector_runs["exposure_kg_yr"].to_numpy(),
            detector_runs["n_events"].to_numpy(),
            detector_runs["n_primaries"].to_numpy(),
        )
        if combined is None:
            continue
        efficiency = combined["efficiency"]
        stat_unc = combined["efficiency_stat_unc"]
        rows.append(
            {
                "sim_e": int(sim_e),
                "detector": str(detector),
                "detector_type": DET_TYPE_MAP.get(str(detector)[0].upper(), "unknown"),
                "efficiency": efficiency,
                "efficiency_stat_unc": stat_unc,
                "relative_unc": stat_unc / efficiency if efficiency > 0 else None,
                "n_primaries": int(detector_runs["n_primaries"].sum()),
                "n_events": int(detector_runs["n_events"].sum()),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "sim_e": pl.Int64,
            "detector": pl.String,
            "detector_type": pl.String,
            "efficiency": pl.Float64,
            "efficiency_stat_unc": pl.Float64,
            "relative_unc": pl.Float64,
            "n_primaries": pl.Int64,
            "n_events": pl.Int64,
        },
    )


def _log_energy_report(rows: pl.DataFrame, *, top: int, target: float) -> None:
    """Log the detectors of one energy with the largest uncertainties."""
    energy = rows["sim_e"][0]
    above = rows.filter(pl.col("relative_unc") > target).height
    logger.info(
        "\n%d keV: %d of %d detectors above the %.2f%% target",
        energy,
        above,
        rows.height,
        100 * target,
    )
    worst = rows["relative_unc"].max()
    if above and worst is not None:
        # The relative uncertainty falls as 1 / sqrt(primaries).
        logger.info(
            "  Worst detector needs about %.1fx its simulated primaries",
            (worst / target) ** 2,
        )
    for title, column in (
        ("relative", "relative_unc"),
        ("absolute", "efficiency_stat_unc"),
    ):
        logger.info("  Largest %s statistical uncertainty:", title)
        ranked = rows.sort(column, descending=True, nulls_last=True).head(top)
        for row in ranked.iter_rows(named=True):
            relative = row["relative_unc"]
            logger.info(
                "    %s (%s): efficiency = %.5f +- %.5f (%s), primaries = %d",
                row["detector"],
                row["detector_type"],
                row["efficiency"],
                row["efficiency_stat_unc"],
                "n/a" if relative is None else f"{100 * relative:.2f}%",
                row["n_primaries"],
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Report the statistical uncertainty of the simulated efficiencies."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "Rank detectors by the statistical uncertainty of their simulated "
            "efficiency, using the table written by the efficiencies stage."
        )
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--interaction",
        required=True,
        help="Interaction name (e.g. axio-electric, dark-compton)",
    )
    parser.add_argument(
        "--selection", default="all", help="Efficiency selection (default: all)"
    )
    parser.add_argument(
        "--top", type=int, default=5, help="Detectors listed per energy (default: 5)"
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.01,
        help="Target relative statistical uncertainty (default: 0.01)",
    )
    args = parser.parse_args(argv)
    if args.top < 1 or args.target <= 0:
        parser.error("--top must be at least 1 and --target must be positive")

    try:
        config = load_analysis_config(args.config)
    except Exception as exc:
        parser.error(f"Failed to load config: {exc}")
    if args.interaction not in config.interactions:
        parser.error(
            f"Unknown interaction {args.interaction!r}; "
            f"configured: {sorted(config.interactions)}"
        )

    path = efficiency_table_path(config.paths.dictionaries_root, args.interaction)
    if not path.exists():
        parser.error(
            f"Efficiency table not found at {path}; run the efficiencies stage first"
        )

    stats = efficiency_stat_table(load_efficiency_table(path), args.selection)
    if stats.is_empty():
        parser.error(f"No valid {args.selection!r} efficiency in {path}")

    logger.info("Efficiency statistics for %s (%s)", args.interaction, path)
    for _, rows in stats.group_by("sim_e", maintain_order=True):
        _log_energy_report(rows, top=args.top, target=args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

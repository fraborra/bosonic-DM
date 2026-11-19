# Copyright (C) 2026 Francesco Borra
#

r"""Expected sensitivity (Brazil plot) from the saved background fits.

Loads ``B_hat`` of every group (``bhat.parquet`` written by
``save_background_fit``), the signal efficiency table and the resolution
inputs, draws background-only pseudo-spectra and computes the distribution of
the 90% credible upper limit on the signal rate at every mass. No limit on the
real data is computed.

Example (from the repository root)::

    PYTHONPATH=src pixi run python3 scripts/expected_sensitivity.py \\
        --fits-dir data/v8fits/background \\
        --fit-subdir no_linear_shape_sd0.5 \\
        --inputs data/inputs/dictionaries \\
        --efficiency <sim_root>/axio-electric_efficiency.parquet \\
        --masses 300,500,800 --n-toys 200 --outdir data/v8fits/sensitivity/smoke

Outputs in ``--outdir``: ``limits_per_toy.parquet``,
``expected_limits.parquet``, ``expected_limits.yaml``,
``expected_limits.png`` and one ``window_<mass>keV.png`` per
``--window-plot-masses`` entry.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import polars as pl
from dbetto import Props

from bosonic_dm.efficiency import load_efficiency_table
from bosonic_dm.fitting.sensitivity import expected_limits
from bosonic_dm.fitting.signal_template import (
    BackgroundPrediction,
    build_signal_window,
    group_efficiency,
    load_background_prediction,
)
from bosonic_dm.plotting.sensitivity import plot_expected_limits
from bosonic_dm.plotting.signal_fit import plot_window_fit
from bosonic_dm.yaml_io import write_yaml

logger = logging.getLogger("expected_sensitivity")


def parse_masses(text: str) -> np.ndarray:
    """Parse ``"start:stop:step"`` (stop included) or ``"m1,m2,..."``."""
    if ":" in text:
        start, stop, step = (float(part) for part in text.split(":"))
        if step <= 0 or stop < start:
            msg = f"Invalid mass range {text!r}."
            raise argparse.ArgumentTypeError(msg)
        return np.round(np.arange(start, stop + 0.5 * step, step), 9)
    return np.array([float(part) for part in text.split(",") if part.strip()])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--fits-dir",
        type=Path,
        required=True,
        help="Directory with one sub-directory per group (spaces as underscores).",
    )
    parser.add_argument(
        "--fit-subdir",
        default="no_linear_shape_sd0.5",
        help="Fit variant inside each group directory.",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        required=True,
        help="Directory with eres_dict.yaml and detector-grouping/groups_dict.yaml.",
    )
    parser.add_argument(
        "--efficiency", type=Path, required=True, help="Efficiency Parquet table."
    )
    parser.add_argument(
        "--groups", nargs="+", default=None, help="Groups to use (default: all)."
    )
    parser.add_argument("--selection", default="all", help="Efficiency selection.")
    parser.add_argument(
        "--masses", type=parse_masses, default=parse_masses("123:1022:1")
    )
    parser.add_argument("--n-toys", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--credibility", type=float, default=0.9)
    parser.add_argument("--half-width", type=float, default=12.5)
    parser.add_argument("--exposure-key", default="expo")
    parser.add_argument(
        "--window-plot-masses",
        type=parse_masses,
        default=parse_masses("500"),
        help="Masses at which to plot counts, B_hat and B_hat + R90 S.",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args(argv)


def usable_masses(
    masses: np.ndarray,
    backgrounds: dict[str, BackgroundPrediction],
    efficiency_ranges: dict[str, tuple[float, float]],
    half_width_keV: float,
) -> np.ndarray:
    """Drop masses outside the simulated range or too close to the fit edges."""
    edges = next(iter(backgrounds.values())).bin_edges
    low = max(
        [edges[0] + half_width_keV] + [lo for lo, _ in efficiency_ranges.values()]
    )
    high = min(
        [edges[-1] - half_width_keV] + [hi for _, hi in efficiency_ranges.values()]
    )
    keep = (masses >= low) & (masses <= high)
    if not keep.all():
        logger.warning(
            "Dropped %d masses outside [%g, %g] keV (simulated range and fit range).",
            int((~keep).sum()),
            low,
            high,
        )
    return masses[keep]


def main(argv: Sequence[str] | None = None) -> None:
    """Run the expected-sensitivity study and write its outputs."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    eres_dict = Props.read_from(str(args.inputs / "eres_dict.yaml"))
    groups_dict = Props.read_from(
        str(args.inputs / "detector-grouping" / "groups_dict.yaml")
    )
    groups = args.groups or list(groups_dict)
    groups_dict = {name: groups_dict[name] for name in groups}

    backgrounds = {
        name: load_background_prediction(
            args.fits_dir / name.replace(" ", "_") / args.fit_subdir, group=name
        )
        for name in groups
    }
    failed = [name for name, b in backgrounds.items() if not b.passed_diagnostics]
    if failed:
        logger.warning("Background fits that failed their diagnostics: %s", failed)

    efficiencies = group_efficiency(
        load_efficiency_table(args.efficiency), groups_dict, selection=args.selection
    )
    ranges = {
        name: (float(e.energies_keV[0]), float(e.energies_keV[-1]))
        for name, e in efficiencies.items()
    }
    masses = usable_masses(args.masses, backgrounds, ranges, args.half_width)
    if masses.size == 0:
        msg = "No mass left to analyse."
        raise SystemExit(msg)
    logger.info(
        "%d masses in [%g, %g] keV, %d toys, groups %s.",
        masses.size,
        masses.min(),
        masses.max(),
        args.n_toys,
        groups,
    )

    window_options = {
        "eres_dict": eres_dict,
        "groups_dict": groups_dict,
        "group_efficiencies": efficiencies,
        "half_width_keV": args.half_width,
        "exposure_key": args.exposure_key,
    }
    result = expected_limits(
        masses,
        backgrounds,
        n_toys=args.n_toys,
        base_seed=args.seed,
        credibility=args.credibility,
        **window_options,
    )

    paths = {
        "per_toy": args.outdir / "limits_per_toy.parquet",
        "summary": args.outdir / "expected_limits.parquet",
        "record": args.outdir / "expected_limits.yaml",
        "plot": args.outdir / "expected_limits.png",
    }
    result.per_toy.write_parquet(paths["per_toy"])
    result.summary.write_parquet(paths["summary"])
    write_yaml(
        {
            **result.settings,
            "inputs": {
                "fits_dir": str(args.fits_dir),
                "fit_subdir": args.fit_subdir,
                "inputs": str(args.inputs),
                "efficiency": str(args.efficiency),
                "selection": args.selection,
            },
            "efficiency_ranges_keV": ranges,
            "masses_keV": {
                "n": int(masses.size),
                "min": float(masses.min()),
                "max": float(masses.max()),
            },
            "files": {name: path.name for name, path in paths.items()},
        },
        paths["record"],
    )
    plot_expected_limits(
        result.summary,
        title=f"Expected {args.credibility:.0%} upper limit, {args.n_toys} toys",
        output_path=paths["plot"],
    )

    summary = result.summary
    for mass in args.window_plot_masses:
        row = summary.filter((pl.col("mass_keV") - float(mass)).abs() < 1e-6)
        if row.is_empty() or bool(row["all_bins_masked"][0]):
            logger.warning("No limit at %g keV; window plot skipped.", mass)
            continue
        window = build_signal_window(float(mass), backgrounds, **window_options)
        plot_window_fit(
            window,
            float(row["r90_median"][0]),
            rate_label=r"$\hat B + R_{90}^{\mathrm{med}} S$",
            output_path=args.outdir / f"window_{mass:g}keV.png",
        )
    logger.info(
        "Done in %.0f s; outputs in %s.", result.settings["runtime_s"], args.outdir
    )


if __name__ == "__main__":
    main()

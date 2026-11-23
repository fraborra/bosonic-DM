# Copyright (C) 2026 Francesco Borra
#

r"""Expected sensitivity (Brazil plot) from the saved background fits.

Reads, from the pipeline outputs named by an analysis config, ``B_hat`` of
every group (``bhat.parquet`` written by the background-fit stage), the signal
efficiency table and the resolution dictionary. It draws background-only
pseudo-spectra and computes the distribution of the 90% credible upper limit
on the signal rate at every mass. The analysis is blinded: the observed counts
are never fitted or drawn; the window plots show toy 0.

Example (NERSC, from the repository root)::

    PYTHONPATH=src pixi run python3 scripts/expected_sensitivity.py \
        --config configs/nersc.yaml \
        --masses 300,500,800 --n-toys 200 --outdir data/v5/sensitivity/smoke

Outputs in ``--outdir``: ``limits_per_toy.parquet``,
``expected_limits.parquet``, ``expected_limits.yaml``,
``expected_limits.png`` and one ``window_<mass>keV.png`` per
``--window-plot-masses`` entry.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import polars as pl

from bosonic_dm.config import load_analysis_config
from bosonic_dm.fitting.sensitivity import (
    draw_background_toys,
    expected_limits,
    stack_backgrounds,
    window_slice,
)
from bosonic_dm.fitting.signal_template import (
    BackgroundPrediction,
    build_signal_window,
)
from bosonic_dm.pipeline.sensitivity import (
    load_signal_inputs,
    parse_masses,
    usable_masses,
)
from bosonic_dm.plotting.sensitivity import plot_expected_limits
from bosonic_dm.plotting.signal_fit import plot_window_fit
from bosonic_dm.yaml_io import write_yaml

logger = logging.getLogger("expected_sensitivity")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True, help="Analysis YAML.")
    parser.add_argument("--interaction", default="axio-electric")
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
        help="Masses at which to plot toy 0, B_hat and B_hat + R90 S.",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args(argv)


def plot_window_toys(
    summary: pl.DataFrame,
    backgrounds: Mapping[str, BackgroundPrediction],
    masses: Sequence[float],
    *,
    seed: int,
    window_options: Mapping,
    outdir: Path,
) -> None:
    """Plot toy 0, ``B_hat`` and ``B_hat`` plus the median limit at some masses.

    Toy 0 is drawn on the full spectrum with the stream used by
    ``expected_limits``, then cut to each window. The observed counts are
    never drawn.
    """
    edges, b_hat = stack_backgrounds(backgrounds)
    toy = draw_background_toys(b_hat, 1, seed)[0]
    for mass in masses:
        row = summary.filter((pl.col("mass_keV") - float(mass)).abs() < 1e-6)
        if row.is_empty() or bool(row["all_bins_masked"][0]):
            logger.warning("No limit at %g keV; window plot skipped.", mass)
            continue
        window = build_signal_window(float(mass), backgrounds, **window_options)
        toy_window = window.with_counts(toy[:, window_slice(edges, window.bin_edges)])
        fig = plot_window_fit(
            toy_window,
            float(row["r90_median"][0]),
            counts_label="toy 0",
            rate_label=r"$\hat B + R_{90}^{\mathrm{med}} S$",
            title=f"m = {mass:g} keV, blinded (toy 0)",
            output_path=outdir / f"window_{mass:g}keV.png",
        )
        plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the expected-sensitivity study and write its outputs."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    config = load_analysis_config(args.config)
    inputs = load_signal_inputs(
        config,
        groups=args.groups,
        interaction=args.interaction,
        selection=args.selection,
    )
    backgrounds = inputs.backgrounds

    ranges = {
        name: (float(e.energies_keV[0]), float(e.energies_keV[-1]))
        for name, e in inputs.group_efficiencies.items()
    }
    masses = usable_masses(args.masses, backgrounds, ranges, args.half_width)
    if masses.size == 0:
        msg = "No mass left to analyse."
        raise SystemExit(msg)
    logger.info(
        "%d masses in [%g, %g] keV, %d toys, groups %s; blinded.",
        masses.size,
        masses.min(),
        masses.max(),
        args.n_toys,
        list(backgrounds),
    )

    window_options = {
        **inputs.window_options,
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
                "config": str(args.config),
                "interaction": args.interaction,
                "selection": args.selection,
                "efficiency_table": str(inputs.efficiency_table),
                "background_fit_dirs": {
                    name: str(path) for name, path in inputs.fit_dirs.items()
                },
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
    fig = plot_expected_limits(
        result.summary,
        title=f"Expected {args.credibility:.0%} upper limit, {args.n_toys} toys",
        output_path=paths["plot"],
    )
    plt.close(fig)

    plot_window_toys(
        result.summary,
        backgrounds,
        args.window_plot_masses,
        seed=args.seed,
        window_options=window_options,
        outdir=args.outdir,
    )
    logger.info(
        "Done in %.0f s; outputs in %s.", result.settings["runtime_s"], args.outdir
    )


if __name__ == "__main__":
    main()

# Copyright (C) 2026 Francesco Borra
#

r"""Stan versus grid-posterior limits on the same background-only toys.

Draws the toys of the expected-sensitivity study (same seed, same stream),
fits each with the Stan signal model at a set of DM masses and compares the
Stan 90% credible upper limit with the grid-posterior one. The masses can be
split across SLURM array jobs; every job writes one parquet per mass and the
final comparison reads them all. The analysis is blinded: the observed counts
are never used.

Example (NERSC, from the repository root)::

    sbatch --array=0-3 scripts/submit_stan_comparison.sh \
        --config configs/nersc.yaml \
        --grid-per-toy data/v5/sensitivity/full/limits_per_toy.parquet \
        --outdir data/v5/sensitivity/stan_check

More toys can be added later without refitting the saved ones: toy ``k`` and
its Stan seed depend only on ``k`` (and the mass), so a run with
``--n-toys 1000 --toys 200:1000`` in the same ``--outdir`` fits toys 200-999
and the comparison then uses all 1000::

    sbatch --array=0-7 scripts/submit_stan_comparison.sh \
        --config configs/nersc.yaml --n-toys 1000 --toys 200:1000 \
        --grid-per-toy data/v5/sensitivity/full/limits_per_toy.parquet \
        --outdir data/v5/sensitivity/stan_check

Outputs in ``--outdir``: ``per_mass/<mass>keV_t<first>-<last>.parquet`` (one
per mass and block of toys; ``<mass>keV.parquet`` of earlier versions is read
too), ``stan_per_toy.parquet``, ``comparison.parquet``, ``comparison.png`` and
``comparison.yaml``.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from bosonic_dm.config import load_analysis_config
from bosonic_dm.fitting.sensitivity import (
    agreement_checks,
    compare_stan_grid,
    draw_background_toys,
    expected_limits,
    stack_backgrounds,
    stan_limits,
    window_slice,
)
from bosonic_dm.fitting.signal_fit import SIGNAL_STAN_FILE
from bosonic_dm.fitting.signal_template import build_signal_window
from bosonic_dm.pipeline.sensitivity import (
    SignalInputs,
    load_signal_inputs,
    parse_masses,
    usable_masses,
)
from bosonic_dm.plotting.sensitivity import plot_stan_grid_comparison
from bosonic_dm.yaml_io import write_yaml

if TYPE_CHECKING:
    import cmdstanpy

logger = logging.getLogger("stan_grid_comparison")

# Named mass grids for --masses (keV, ``start:stop:step`` with stop included).
# "coarse": the first 37-mass check. "fine": every 5 keV from 123 to 400 keV,
# every 1 keV within 8 keV of the masked gamma lines (238.6, 295.2, 351.9,
# 911.2, 969.0 keV) plus 10, 13 and 17 keV away, and every 2 keV around the
# Ar-39 endpoint (545-585 keV); it contains all the "coarse" masses (204 total).
MASS_PRESETS = {
    "coarse": "123,150,175,200:400:10,450:1000:50,1020",
    "fine": (
        "123,125:400:5,200:400:10,"
        "222,226,229,231:247:1,249,252,256,"
        "278,282,285,287:303:1,305,308,312,"
        "335,339,342,344:360:1,362,365,369,"
        "450:1000:50,520:610:5,545:585:2,"
        "894,898,901,903:919:1,921,924,928,"
        "952,956,959,961:977:1,979,982,986,1020"
    ),
}
DEFAULT_MASSES = "coarse"


def parse_mass_option(text: str) -> np.ndarray:
    """``--masses``: a preset name from ``MASS_PRESETS`` or an explicit list."""
    return parse_masses(MASS_PRESETS.get(text.strip(), text))


def split_masses(masses: np.ndarray, n_jobs: int, job_index: int) -> np.ndarray:
    """Masses of one array job: ``masses[job_index::n_jobs]``.

    Parameters
    ----------
    masses
        All masses to analyse.
    n_jobs
        Number of jobs.
    job_index
        Index of this job, ``0 <= job_index < n_jobs``.

    Returns
    -------
    numpy.ndarray
        The interleaved subset of this job.

    Raises
    ------
    ValueError
        If ``n_jobs`` or ``job_index`` is out of range.
    """
    if n_jobs < 1 or not 0 <= job_index < n_jobs:
        msg = f"Invalid job split: job_index={job_index}, n_jobs={n_jobs}."
        raise ValueError(msg)
    return np.asarray(masses)[job_index::n_jobs]


def default_job_split(env: Mapping[str, str] | None = None) -> tuple[int, int]:
    """``(n_jobs, job_index)`` from the SLURM array variables, else ``(1, 0)``."""
    env = os.environ if env is None else env
    if "SLURM_ARRAY_TASK_COUNT" in env and "SLURM_ARRAY_TASK_ID" in env:
        first = int(env.get("SLURM_ARRAY_TASK_MIN", "0"))
        return int(env["SLURM_ARRAY_TASK_COUNT"]), int(
            env["SLURM_ARRAY_TASK_ID"]
        ) - first
    return 1, 0


def parse_toy_range(text: str) -> tuple[int, int]:
    """Parse ``start:stop`` (stop excluded) into ``(start, stop)``.

    Parameters
    ----------
    text
        Toy range, e.g. ``"200:1000"`` for toys 200 to 999.

    Returns
    -------
    tuple of int
        ``(start, stop)``.

    Raises
    ------
    argparse.ArgumentTypeError
        If the range is malformed, negative or empty.
    """
    try:
        start, stop = (int(part) for part in text.split(":"))
    except ValueError as err:
        msg = f"Invalid toy range {text!r}; expected start:stop."
        raise argparse.ArgumentTypeError(msg) from err
    if start < 0 or stop <= start:
        msg = f"Invalid toy range {text!r}; need 0 <= start < stop."
        raise argparse.ArgumentTypeError(msg)
    return start, stop


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True, help="Analysis YAML.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--masses",
        type=parse_mass_option,
        default=parse_mass_option(DEFAULT_MASSES),
        help=(
            f"Preset ({', '.join(MASS_PRESETS)}; default {DEFAULT_MASSES}) or a "
            "comma list of masses and start:stop:step ranges in keV."
        ),
    )
    parser.add_argument(
        "--n-toys",
        type=int,
        default=200,
        help="Total toys per mass; the comparison is complete with toys 0 to N-1.",
    )
    parser.add_argument(
        "--toys",
        type=parse_toy_range,
        default=None,
        help="Toys to fit in this run, start:stop with stop excluded "
        "(default 0:N); saved toys are skipped unless --overwrite.",
    )
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--credibility", type=float, default=0.9)
    parser.add_argument("--half-width", type=float, default=12.5)
    parser.add_argument("--exposure-key", default="expo")
    parser.add_argument("--interaction", default="axio-electric")
    parser.add_argument("--selection", default="all", help="Efficiency selection.")
    parser.add_argument(
        "--groups", nargs="+", default=None, help="Groups to use (default: all)."
    )
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--iter-warmup", type=int, default=1000)
    parser.add_argument("--iter-sampling", type=int, default=2000)
    parser.add_argument(
        "--adapt-delta", type=float, default=0.9, help="Stan target acceptance."
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--job-index", type=int, default=None)
    parser.add_argument(
        "--grid-per-toy",
        type=Path,
        default=None,
        help="limits_per_toy.parquet of expected_sensitivity.py, to cross-check.",
    )
    parser.add_argument("--stan-exe", type=Path, default=None, help="Compiled model.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args(argv)
    if args.toys is None:
        args.toys = (0, args.n_toys)
    if args.toys[1] > args.n_toys:
        parser.error(f"--toys {args.toys[0]}:{args.toys[1]} goes beyond --n-toys.")
    default_jobs, default_index = default_job_split()
    if args.n_jobs is None:
        args.n_jobs = default_jobs
    if args.job_index is None:
        args.job_index = default_index
    return args


def _mass_files(outdir: Path, mass: float) -> list[Path]:
    """Saved parquet files of one mass, including the old one-file name."""
    per_mass = outdir / "per_mass"
    old = per_mass / f"{mass:g}keV.parquet"
    chunks = sorted(per_mass.glob(f"{mass:g}keV_t*.parquet"))
    return [old, *chunks] if old.is_file() else chunks


def _chunk_file(outdir: Path, mass: float, toys: Sequence[int]) -> Path:
    return outdir / "per_mass" / f"{mass:g}keV_t{min(toys)}-{max(toys)}.parquet"


def _saved_toys(files: Sequence[Path]) -> set[int]:
    return {
        int(toy)
        for path in files
        for toy in pl.read_parquet(path, columns=["toy"])["toy"]
    }


def _write_atomic(table: pl.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    table.write_parquet(tmp)
    tmp.replace(path)


def _drop_toys(files: Sequence[Path], toys: Sequence[int], keep: Path) -> None:
    """Remove ``toys`` from the saved files other than ``keep``.

    Run before ``keep`` is written: if the job stops in between, the removed
    toys are missing, not duplicated, and the next run fits them again.
    """
    for path in files:
        if path == keep:
            continue
        table = pl.read_parquet(path)
        kept = table.filter(~pl.col("toy").is_in(list(toys)))
        if kept.height == table.height:
            continue
        if kept.height == 0:
            path.unlink()
        else:
            _write_atomic(kept, path)


def _check_against_grid(
    per_toy: pl.DataFrame, grid_path: Path, masses: Sequence[float], n_toys: int
) -> None:
    """Raise ``SystemExit`` if the recomputed grid limits differ from the table."""
    saved = pl.read_parquet(grid_path)
    for mass in masses:
        ours = per_toy.filter((pl.col("mass_keV") - mass).abs() < 1e-6).sort("toy")
        theirs = saved.filter(
            ((pl.col("mass_keV") - mass).abs() < 1e-6) & (pl.col("toy") < n_toys)
        ).sort("toy")
        if ours.height != theirs.height or not np.allclose(
            ours["r90"].to_numpy(),
            theirs["r90"].to_numpy(),
            rtol=1e-9,
            equal_nan=True,
        ):
            msg = f"Grid R90 at {mass:g} keV differs from {grid_path}."
            raise SystemExit(msg)


def fit_stage(
    args: argparse.Namespace,
    inputs: SignalInputs,
    masses: np.ndarray,
    model: cmdstanpy.CmdStanModel,
) -> None:
    """Fit this job's masses with Stan; one parquet per mass and block of toys."""
    job_masses = split_masses(masses, args.n_jobs, args.job_index)
    requested = list(range(*args.toys))
    plan: dict[float, tuple[list[int], list[Path]]] = {}
    for m in job_masses:
        files = _mass_files(args.outdir, float(m))
        toys_left = (
            requested if args.overwrite else sorted(set(requested) - _saved_toys(files))
        )
        if toys_left:
            plan[float(m)] = (toys_left, files)
    todo = list(plan)
    logger.info(
        "Job %d/%d: %d masses, %d to fit (toys %d to %d).",
        args.job_index,
        args.n_jobs,
        job_masses.size,
        len(todo),
        args.toys[0],
        args.toys[1] - 1,
    )
    if not todo:
        return
    config = load_analysis_config(args.config)
    backgrounds = inputs.backgrounds
    window_options = {
        **inputs.window_options,
        "half_width_keV": args.half_width,
        "exposure_key": args.exposure_key,
    }
    edges, b_hat = stack_backgrounds(backgrounds)
    toys = draw_background_toys(b_hat, args.n_toys, args.seed)
    grid = expected_limits(
        todo,
        backgrounds,
        n_toys=args.n_toys,
        base_seed=args.seed,
        credibility=args.credibility,
        n_bootstrap=0,
        toys=toys,
        **window_options,
    )
    if args.grid_per_toy is not None:
        _check_against_grid(grid.per_toy, args.grid_per_toy, todo, args.n_toys)

    temporary_root = Path(config.paths.temporary_root)
    temporary_root.mkdir(parents=True, exist_ok=True)
    (args.outdir / "per_mass").mkdir(parents=True, exist_ok=True)
    for mass in todo:
        row = grid.summary.filter((pl.col("mass_keV") - mass).abs() < 1e-6)
        if bool(row["all_bins_masked"][0]):
            logger.warning("All bins masked at %g keV; skipped.", mass)
            continue
        start = time.perf_counter()
        toys_left, files = plan[mass]
        window = build_signal_window(mass, backgrounds, **window_options)
        counts = toys[toys_left][:, :, window_slice(edges, window.bin_edges)]
        stan = stan_limits(
            window,
            counts,
            toys_left,
            model,
            base_seed=args.seed,
            credibility=args.credibility,
            chains=args.chains,
            iter_warmup=args.iter_warmup,
            iter_sampling=args.iter_sampling,
            adapt_delta=args.adapt_delta,
            temporary_root=temporary_root,
        )
        grid_mass = grid.per_toy.filter(
            (pl.col("mass_keV") - mass).abs() < 1e-6
        ).select("toy", "r90")
        table = stan.join(grid_mass, on="toy", how="left").with_columns(
            pl.lit(bool(row["near_masked_line"][0])).alias("near_masked_line")
        )
        path = _chunk_file(args.outdir, mass, toys_left)
        if args.overwrite:
            _drop_toys(files, toys_left, keep=path)
        _write_atomic(table, path)
        logger.info(
            "m = %g keV done in %.0f s (%d toys).",
            mass,
            time.perf_counter() - start,
            len(toys_left),
        )


def compare_stage(
    args: argparse.Namespace, inputs: SignalInputs, masses: np.ndarray
) -> None:
    """Compare all per-mass results present and write the summary outputs."""
    files = sorted((args.outdir / "per_mass").glob("*.parquet"))
    if not files:
        logger.warning("No per-mass file in %s; nothing to compare.", args.outdir)
        return
    per_toy = pl.concat([pl.read_parquet(f) for f in files])
    duplicated = per_toy.group_by("mass_keV", "toy").len().filter(pl.col("len") > 1)
    if duplicated.height:
        masses_hit = sorted(duplicated["mass_keV"].unique().to_list())
        msg = (
            f"{duplicated.height} toys are saved more than once in "
            f"{args.outdir / 'per_mass'} (masses {masses_hit} keV); refit them "
            "with --overwrite."
        )
        raise SystemExit(msg)
    beyond = per_toy.filter(pl.col("toy") >= args.n_toys).height
    if beyond:
        logger.info("Ignoring %d fits of toys >= --n-toys %d.", beyond, args.n_toys)
    per_toy = per_toy.filter(pl.col("toy") < args.n_toys).sort("mass_keV", "toy")
    if per_toy.height == 0:
        logger.warning("No toy below --n-toys %d; nothing to compare.", args.n_toys)
        return
    comparison = compare_stan_grid(per_toy, seed=0)
    checks = agreement_checks(comparison, per_toy)

    per_toy.write_parquet(args.outdir / "stan_per_toy.parquet")
    comparison.write_parquet(args.outdir / "comparison.parquet")
    fig = plot_stan_grid_comparison(
        comparison,
        per_toy,
        title=f"Stan vs grid, {args.n_toys} toys",
        output_path=args.outdir / "comparison.png",
    )
    plt.close(fig)

    toys_per_mass = {
        float(m): int(n) for m, n in per_toy.group_by("mass_keV").len().iter_rows()
    }
    present = sorted(toys_per_mass)

    def n_saved(mass: float) -> int:
        return next((n for m, n in toys_per_mass.items() if abs(m - mass) < 1e-6), 0)

    incomplete = [float(m) for m in masses if n_saved(float(m)) < args.n_toys]
    complete = not incomplete
    write_yaml(
        {
            "settings": {
                "n_toys": args.n_toys,
                "seed": args.seed,
                "credibility": args.credibility,
                "half_width_keV": args.half_width,
                "exposure_key": args.exposure_key,
                "sampler": {
                    "chains": args.chains,
                    "iter_warmup": args.iter_warmup,
                    "iter_sampling": args.iter_sampling,
                    "adapt_delta": args.adapt_delta,
                },
            },
            "inputs": {
                "config": str(args.config),
                "interaction": args.interaction,
                "selection": args.selection,
                "efficiency_table": str(inputs.efficiency_table),
                "background_fit_dirs": {
                    name: str(path) for name, path in inputs.fit_dirs.items()
                },
                "files_sha256": {
                    name: dict(bg.files_sha256)
                    for name, bg in inputs.backgrounds.items()
                },
            },
            "masses_requested": [float(m) for m in masses],
            "masses_present": present,
            "masses_incomplete": incomplete,
            "complete": bool(complete),
            "n_jobs": args.n_jobs,
            "job_index": args.job_index,
            "agreement": checks,
        },
        args.outdir / "comparison.yaml",
    )
    logger.info("Comparison written (%d masses, complete=%s).", len(present), complete)


def main(argv: Sequence[str] | None = None) -> None:
    """Run this job's Stan fits and the comparison of everything present."""
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
    ranges = {
        name: (float(e.energies_keV[0]), float(e.energies_keV[-1]))
        for name, e in inputs.group_efficiencies.items()
    }
    masses = usable_masses(args.masses, inputs.backgrounds, ranges, args.half_width)
    if masses.size == 0:
        msg = "No mass left to analyse."
        raise SystemExit(msg)

    if not args.compare_only:
        from cmdstanpy import CmdStanModel  # noqa: PLC0415

        model = (
            CmdStanModel(exe_file=str(args.stan_exe))
            if args.stan_exe is not None
            else CmdStanModel(stan_file=str(SIGNAL_STAN_FILE))
        )
        fit_stage(args, inputs, masses, model)
    compare_stage(args, inputs, masses)


if __name__ == "__main__":
    main()

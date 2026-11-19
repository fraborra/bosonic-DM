# Copyright (C) 2026 Francesco Borra
#

r"""Check the fixed-background signal fit at one DM mass, blinded by default.

The script reads the background fits, the resolution dictionary and the
efficiency table named by an analysis config, and builds the signal window at
one mass. Without ``--unblind`` the observed counts in the window are never
used. It then:

1. logs the response inputs of every group (exposure, efficiency, FWHM,
   window sums of ``B_hat`` and of the signal template);
2. computes the 90% upper limit on an Asimov dataset (counts = ``B_hat``) with
   the grid posterior; Stan needs integer counts, so Asimov is grid only;
3. draws one Poisson toy from ``B_hat`` (plus ``--inject`` times the signal),
   fits it with Stan and with the grid posterior, and compares the limits;
4. writes ``summary.yaml`` and ``rate_draws.npy`` (Stan fit of the toy),
   ``checks.yaml`` and ``window.png`` to ``--outdir``.

``--unblind`` fits the observed counts instead of the toy.

Example (NERSC)::

    PYTHONPATH=src pixi run python3 scripts/signal_fit_one_mass.py \
        --config configs/nersc.yaml --mass 500 --outdir tmp/signal_500
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from bosonic_dm.config import AnalysisConfig, load_analysis_config
from bosonic_dm.fitting.signal_fit import (
    SIGNAL_STAN_FILE,
    build_signal_stan_data,
    rate_posterior_quantiles,
    run_signal_fit,
    save_signal_fit,
    suggest_rate_upper,
    summarize_signal_fit,
)
from bosonic_dm.fitting.signal_template import (
    SignalWindow,
    build_signal_window,
)
from bosonic_dm.pipeline.sensitivity import load_signal_inputs
from bosonic_dm.plotting.signal_fit import plot_window_fit
from bosonic_dm.yaml_io import write_yaml

logger = logging.getLogger("signal_fit_one_mass")

# Stan and grid limits agree if they differ by less than this many Monte
# Carlo standard errors of the Stan quantile, as in tests/test_stan_signal.py.
# Over 12 seeds on one toy the Stan R90 was unbiased, but its spread was about
# 1.5 times the ArviZ quantile MCSE, so 3 MCSE would fail too often.
AGREEMENT_MCSE = 4.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True, help="Analysis YAML.")
    parser.add_argument("--mass", type=float, required=True, help="DM mass [keV].")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--interaction", default="axio-electric")
    parser.add_argument("--selection", default="all", help="Efficiency selection.")
    parser.add_argument(
        "--groups", nargs="+", default=None, help="Groups to use (default: all)."
    )
    parser.add_argument("--half-width", type=float, default=12.5)
    parser.add_argument("--credibility", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20261119)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--inject",
        type=float,
        default=0.0,
        help="Signal rate added to the toy, in counts/(kg yr).",
    )
    mode.add_argument(
        "--unblind",
        action="store_true",
        help="Fit the observed counts instead of a toy.",
    )
    args = parser.parse_args(argv)
    if args.inject < 0:
        parser.error("--inject must be nonnegative.")
    return args


def load_window(
    config: AnalysisConfig,
    mass_keV: float,
    *,
    groups: Sequence[str] | None = None,
    interaction: str = "axio-electric",
    selection: str = "all",
    half_width_keV: float = 12.5,
) -> SignalWindow:
    """Build the signal window at one mass from the pipeline outputs."""
    inputs = load_signal_inputs(
        config, groups=groups, interaction=interaction, selection=selection
    )
    return build_signal_window(
        mass_keV,
        inputs.backgrounds,
        **inputs.window_options,
        half_width_keV=half_width_keV,
    )


def log_inputs(window: SignalWindow) -> None:
    """Log the response inputs of every group; never the observed counts."""
    logger.info(
        "%-12s %9s %7s %9s %9s %6s %5s %9s %9s",
        "group",
        "X[kg yr]",
        "eff",
        "FWHM[keV]",
        "+-",
        "cont.",
        "bins",
        "sum B_hat",
        "sum S",
    )
    for name, inputs in window.groups.items():
        used = inputs.use_bin.astype(bool)
        logger.info(
            "%-12s %9.3f %7.4f %9.3f %9.3f %6.3f %5d %9.2f %9.3f",
            name,
            inputs.exposure_kg_yr,
            inputs.efficiency,
            inputs.fwhm_keV,
            inputs.fwhm_unc_keV,
            inputs.window_containment,
            int(used.sum()),
            float(inputs.b_hat[used].sum()),
            float(inputs.signal_template[used].sum()),
        )


def check_mass(
    window: SignalWindow,
    model: Any,
    outdir: Path,
    *,
    seed: int,
    inject_rate: float = 0.0,
    unblind: bool = False,
    credibility: float = 0.9,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit an Asimov dataset and one toy (or the data) at one mass.

    Parameters
    ----------
    window
        Signal window with the observed counts, from ``load_window``.
    model
        Compiled ``stan/signal.stan`` (a ``cmdstanpy.CmdStanModel``).
    outdir
        Output directory.
    seed
        Seed of the toy and of the Stan sampler.
    inject_rate
        Signal rate added to the toy, in counts/(kg yr).
    unblind
        Fit the observed counts instead of a toy.
    credibility
        Credibility of the upper limit.
    provenance
        Extra provenance written to ``summary.yaml``.

    Returns
    -------
    dict
        The content of ``checks.yaml``.
    """
    if unblind and inject_rate:
        msg = "inject_rate needs a toy; it cannot be used with unblind."
        raise ValueError(msg)
    outdir.mkdir(parents=True, exist_ok=True)
    b_hat = window.stacked("b_hat")
    template = window.stacked("signal_template")
    use_bin = window.stacked("use_bin").astype(bool)

    asimov_limit = float(
        rate_posterior_quantiles(b_hat, b_hat, template, use_bin, (credibility,))[0, 0]
    )

    if unblind:
        dataset: dict[str, Any] = {"kind": "observed"}
        fit_window = window
    else:
        rng = np.random.default_rng(seed)
        counts = rng.poisson(b_hat + inject_rate * template)
        dataset = {"kind": "toy", "seed": seed, "injected_rate": inject_rate}
        fit_window = window.with_counts(counts)

    stan_data = build_signal_stan_data(fit_window, suggest_rate_upper(fit_window))
    fit = run_signal_fit(model, stan_data, seed=seed)
    summary = summarize_signal_fit(fit, stan_data, credibility=credibility)
    grid_median, grid_limit = (
        float(value)
        for value in rate_posterior_quantiles(
            fit_window.stacked("counts"), b_hat, template, use_bin, (0.5, credibility)
        )[0]
    )

    mcse = summary.rate["upper_limit_mcse"]
    difference = summary.upper_limit - grid_limit
    agree = bool(np.isfinite(mcse) and abs(difference) <= AGREEMENT_MCSE * mcse)
    checks: dict[str, Any] = {
        "mass_keV": window.mass_keV,
        "dataset": dataset,
        "credibility": credibility,
        "rate_units": "counts / (kg yr)",
        "asimov_upper_limit_grid": asimov_limit,
        "stan": {
            "upper_limit": summary.upper_limit,
            "upper_limit_mcse": mcse,
            "median": summary.rate["median"],
            "mean": summary.rate["mean"],
            "sd": summary.rate["sd"],
            "passed_diagnostics": summary.passed,
        },
        "grid": {"upper_limit": grid_limit, "median": grid_median},
        "stan_minus_grid_in_mcse": (
            float(difference / mcse) if mcse > 0 else float("nan")
        ),
        "stan_grid_agree": agree,
    }
    if inject_rate:
        checks["injection_pull"] = float(
            (summary.rate["mean"] - inject_rate) / summary.rate["sd"]
        )

    save_signal_fit(
        fit,
        summary,
        fit_window,
        stan_data,
        outdir,
        provenance={**dict(provenance or {}), "dataset": dataset},
    )
    write_yaml(checks, outdir / "checks.yaml")
    label = "data" if unblind else "toy"
    fig = plot_window_fit(
        fit_window,
        summary.upper_limit,
        counts_label=label,
        rate_label=r"$\hat B + R_{90} S$ (Stan, " + label + ")",
        title=f"m = {window.mass_keV:g} keV, {label}, "
        f"R90 = {summary.upper_limit:.3g} / (kg yr)",
        output_path=outdir / "window.png",
    )
    plt.close(fig)
    return checks


def log_checks(checks: Mapping[str, Any]) -> None:
    """Log the limits and the pass/fail of each check."""
    stan, grid = checks["stan"], checks["grid"]
    logger.info(
        "Asimov R90 (grid): %.4g counts/(kg yr)", checks["asimov_upper_limit_grid"]
    )
    logger.info(
        "%s R90: Stan %.4g +- %.2g (MCSE), grid %.4g; difference %.2f MCSE -> %s",
        checks["dataset"]["kind"],
        stan["upper_limit"],
        stan["upper_limit_mcse"],
        grid["upper_limit"],
        checks["stan_minus_grid_in_mcse"],
        "agree" if checks["stan_grid_agree"] else "DISAGREE",
    )
    logger.info(
        "Stan diagnostics: %s", "passed" if stan["passed_diagnostics"] else "FAILED"
    )
    if "injection_pull" in checks:
        logger.info(
            "Injected %.4g, posterior mean %.4g +- %.2g (pull %.2f)",
            checks["dataset"]["injected_rate"],
            stan["mean"],
            stan["sd"],
            checks["injection_pull"],
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the one-mass check and write its outputs."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    from cmdstanpy import CmdStanModel  # noqa: PLC0415

    config = load_analysis_config(args.config)
    window = load_window(
        config,
        args.mass,
        groups=args.groups,
        interaction=args.interaction,
        selection=args.selection,
        half_width_keV=args.half_width,
    )
    logger.info(
        "m = %g keV, window [%g, %g] keV, %s",
        window.mass_keV,
        window.bin_edges[0],
        window.bin_edges[-1],
        "UNBLINDED: observed counts" if args.unblind else "blinded: toy counts",
    )
    log_inputs(window)
    checks = check_mass(
        window,
        CmdStanModel(stan_file=str(SIGNAL_STAN_FILE)),
        args.outdir,
        seed=args.seed,
        inject_rate=args.inject,
        unblind=args.unblind,
        credibility=args.credibility,
        provenance={
            "config": str(args.config),
            "interaction": args.interaction,
            "selection": args.selection,
        },
    )
    log_checks(checks)
    logger.info("Outputs in %s", args.outdir)


if __name__ == "__main__":
    main()

# Copyright (C) 2025 Francesco Borra
#

"""Command-line wrapper around the Python API."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from bosonic_dm.config import AnalysisConfig, load_analysis_config
from bosonic_dm.pipeline import run_background_analysis, run_simulation_analysis
from bosonic_dm.pipeline.background import BACKGROUND_DEFAULT_STAGES
from bosonic_dm.pipeline.simulation import SIMULATION_DEFAULT_STAGES

logger = logging.getLogger(__name__)

# Third-party loggers that produce noisy INFO output (e.g. during GDML
# parsing) which breaks tqdm progress bars.  Set them to WARNING.
_NOISY_LOGGERS = (
    "pyg4ometry",
    "pyg4ometry.gdml",
    "pyg4ometry.gdml.Reader",
)


def _print_banner(
    command: str,
    config: AnalysisConfig,
    *,
    interactions: Sequence[str] | None = None,
    stages: Sequence[str],
    overwrite: bool,
) -> None:
    """Log a startup summary banner."""
    lines: list[str] = []
    lines.append(f"  Command       : {command}")
    if interactions:
        lines.append(f"  Interactions  : {', '.join(interactions)}")
    energies_str = ", ".join(str(e) for e in config.energies_keV)
    lines.append(f"  Energies [keV]: {energies_str}")
    lines.append(f"  Stages        : {' → '.join(stages)}")
    lines.append(f"  Overwrite     : {overwrite}")
    lines.append(f"  Save plots    : {config.output.save_plots}")
    lines.append(f"  Manifest      : {config.output.write_manifest}")
    lines.append(f"  Data root     : {config.paths.data_root}")
    lines.append(f"  LAr veto      : {config.apply_lar_veto}")
    lines.append(f"  Selections    : {', '.join(config.selections)}")
    lines.append(f"  FEP ½-width   : {config.fep_window.half_width_fwhm} FWHM")

    width = max(len(line) for line in lines) + 2
    border = "━" * (width + 2)

    banner = f"\n┏{border}┓\n"
    banner += f"┃  {'bosonic-DM pipeline':^{width}}┃\n"
    banner += f"┣{border}┫\n"
    for line in lines:
        banner += f"┃ {line:<{width}} ┃\n"
    banner += f"┗{border}┛"
    logger.info(banner)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="bosonic-DM analysis pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Global options
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--config", type=Path, required=True, help="Path to configuration YAML file"
    )
    parent_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing products"
    )
    parent_parser.add_argument("--stage", nargs="+", help="Specific stages to run")

    # Simulation subcommand
    sim_parser = subparsers.add_parser(
        "simulation", parents=[parent_parser], help="Run simulation analysis"
    )
    sim_parser.add_argument(
        "--interaction",
        required=True,
        help="Interaction name (e.g. axio-electric, dark-compton)",
    )

    # Background subcommand
    subparsers.add_parser(
        "background", parents=[parent_parser], help="Run background analysis"
    )

    # All subcommand
    all_parser = subparsers.add_parser(
        "all", parents=[parent_parser], help="Run all analyses"
    )
    all_parser.add_argument(
        "--interactions",
        nargs="+",
        default=["axio-electric", "dark-compton"],
        help="Interactions to run",
    )

    args = parser.parse_args(argv)

    # Load configuration
    try:
        config = load_analysis_config(args.config)
    except Exception as e:
        parser.error(f"Failed to load config: {e}")

    # Set overwrite if provided
    # The overwrite flag in CLI overrides config
    overwrite = args.overwrite if args.overwrite else None

    if args.command == "simulation":
        stages = args.stage or SIMULATION_DEFAULT_STAGES
        _print_banner(
            "simulation",
            config,
            interactions=[args.interaction],
            stages=stages,
            overwrite=bool(overwrite),
        )
        artifacts = run_simulation_analysis(
            config, interaction=args.interaction, stages=stages, overwrite=overwrite
        )
        logger.info(
            "Simulation artifacts created for %s: %s", args.interaction, artifacts
        )

    elif args.command == "background":
        stages = args.stage or BACKGROUND_DEFAULT_STAGES
        _print_banner(
            "background",
            config,
            stages=stages,
            overwrite=bool(overwrite),
        )
        artifacts = run_background_analysis(config, stages=stages, overwrite=overwrite)
        logger.info("Background artifacts created: %s", artifacts)

    elif args.command == "all":
        sim_stages = args.stage or SIMULATION_DEFAULT_STAGES
        bg_stages = args.stage or BACKGROUND_DEFAULT_STAGES
        _print_banner(
            "all",
            config,
            interactions=args.interactions,
            stages=list(dict.fromkeys([*sim_stages, *bg_stages])),
            overwrite=bool(overwrite),
        )
        # Run simulation for requested interactions
        for interaction in args.interactions:
            run_simulation_analysis(
                config, interaction=interaction, stages=sim_stages, overwrite=overwrite
            )

        # Run background
        run_background_analysis(config, stages=bg_stages, overwrite=overwrite)
        logger.info("All analyses completed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

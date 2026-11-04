# Copyright (C) 2026 Francesco Borra
#

"""Command-line wrapper around the Python API."""

from __future__ import annotations

import argparse
import logging
import textwrap
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
_BANNER_INNER_WIDTH = 76


def _banner_rows(
    command: str,
    config: AnalysisConfig,
    *,
    interactions: Sequence[str] | None,
    stages: Sequence[str],
    overwrite: bool | None,
) -> list[tuple[str, str]]:
    """Build command-specific rows for the startup banner."""
    effective_overwrite = (
        overwrite if overwrite is not None else config.output.overwrite
    )
    rows = [("Command", command)]

    if command in {"simulation", "all"}:
        if interactions:
            rows.append(("Interactions", ", ".join(interactions)))
        rows.extend(
            [
                ("Simulation root", str(config.paths.simulation_root)),
                (
                    "Energies [keV]",
                    ", ".join(str(energy) for energy in config.energies_keV),
                ),
                ("Selections", ", ".join(config.selections)),
                (
                    "FEP half-width",
                    f"{config.fep_window.half_width_fwhm} FWHM",
                ),
                ("Detector groups", str(config.detector_groups)),
            ]
        )

    if command in {"background", "all"}:
        pet_glob = config.background.pet_glob or "from production config"
        energy_ranges = ", ".join(
            f"{lower}-{upper}" for lower, upper in config.background.energy_ranges_keV
        )
        rows.extend(
            [
                ("PET inputs", pet_glob),
                ("Comparison cut", config.background.comparison_cut_profile),
                ("Energy ranges [keV]", energy_ranges),
                (
                    "Bin widths [keV]",
                    ", ".join(str(width) for width in config.background.bin_widths_keV),
                ),
            ]
        )

    rows.extend(
        [
            ("Stages", " → ".join(stages)),
            ("Overwrite", str(effective_overwrite)),
            ("Save plots", str(config.output.save_plots)),
            ("Manifest", str(config.output.write_manifest)),
            ("Data root", str(config.paths.data_root)),
            ("LAr veto", str(config.apply_lar_veto)),
        ]
    )
    return rows


def _render_banner(rows: Sequence[tuple[str, str]]) -> str:
    """Render a width-bounded Unicode banner, wrapping long values."""
    label_width = max(len(label) for label, _ in rows)
    rendered_lines: list[str] = []
    for label, value in rows:
        prefix = f"  {label:<{label_width}}: "
        continuation = " " * len(prefix)
        available_width = _BANNER_INNER_WIDTH - len(prefix)
        wrapped_value = textwrap.wrap(
            str(value),
            width=available_width,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        rendered_lines.append(prefix + wrapped_value[0])
        rendered_lines.extend(
            continuation + wrapped_line for wrapped_line in wrapped_value[1:]
        )

    border = "━" * (_BANNER_INNER_WIDTH + 2)
    banner_lines = [
        f"┏{border}┓",
        f"┃ {'bosonic-DM pipeline':^{_BANNER_INNER_WIDTH}} ┃",
        f"┣{border}┫",
    ]
    banner_lines.extend(f"┃ {line:<{_BANNER_INNER_WIDTH}} ┃" for line in rendered_lines)
    banner_lines.append(f"┗{border}┛")
    return "\n".join(banner_lines)


def _print_banner(
    command: str,
    config: AnalysisConfig,
    *,
    interactions: Sequence[str] | None = None,
    stages: Sequence[str],
    overwrite: bool | None,
) -> None:
    """Log a startup summary banner."""
    rows = _banner_rows(
        command,
        config,
        interactions=interactions,
        stages=stages,
        overwrite=overwrite,
    )
    logger.info("\n%s", _render_banner(rows))


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
            overwrite=overwrite,
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
            overwrite=overwrite,
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
            overwrite=overwrite,
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

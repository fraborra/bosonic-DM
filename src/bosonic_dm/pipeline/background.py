# Copyright (C) 2026 Francesco Borra
#

"""Background-analysis pipeline orchestration."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import dbetto

from bosonic_dm.background import build_background_dataset
from bosonic_dm.config import AnalysisConfig
from bosonic_dm.models import AnalysisArtifacts
from bosonic_dm.pipeline.context import build_analysis_context
from bosonic_dm.plotting.background import (
    plot_background_partition_summary,
    plot_background_spectrum,
)
from bosonic_dm.yaml_io import write_yaml

logger = logging.getLogger(__name__)

BACKGROUND_STAGE_ORDER = ("build-dataset",)
BACKGROUND_DEFAULT_STAGES = BACKGROUND_STAGE_ORDER
BACKGROUND_STAGE_DEPENDENCIES = {"build-dataset": ()}


def resolve_background_stages(stages: Sequence[str]) -> tuple[str, ...]:
    """Expand requested background stages in execution order."""
    unknown = set(stages) - set(BACKGROUND_STAGE_ORDER)
    if unknown:
        msg = f"Unknown background stages: {sorted(unknown)}"
        raise ValueError(msg)

    resolved: set[str] = set()

    def add_with_dependencies(stage: str) -> None:
        for dependency in BACKGROUND_STAGE_DEPENDENCIES[stage]:
            add_with_dependencies(dependency)
        resolved.add(stage)

    for requested_stage in stages:
        add_with_dependencies(requested_stage)

    return tuple(stage for stage in BACKGROUND_STAGE_ORDER if stage in resolved)


def discover_background_inputs(pet_glob: str) -> tuple[Path, ...]:
    """Return existing PET files matching the configured glob in stable order."""
    return tuple(
        sorted(
            path
            for match in Path.glob(pet_glob, recursive=True)
            if (path := Path(match)).is_file()
        )
    )


def background_dataset_path(config: AnalysisConfig) -> Path:
    """Return the deterministic root of the background Parquet dataset."""
    return config.paths.parquet_root / "background"


def background_manifest_path(config: AnalysisConfig) -> Path:
    """Return the deterministic path of the background pipeline manifest."""
    return config.paths.data_root / "background_manifest.yaml"


def _has_parquet_fragments(dataset_path: Path) -> bool:
    """Return whether a dataset directory contains at least one Parquet file."""
    first_fragment = next(dataset_path.rglob("*.parquet"), None)
    return dataset_path.is_dir() and first_fragment is not None


def _warn(artifacts: AnalysisArtifacts, message: str, *args: object) -> None:
    """Log and retain a non-fatal pipeline warning."""
    rendered = message % args if args else message
    logger.warning(rendered)
    artifacts.warnings.append(rendered)


def _write_manifest(
    config: AnalysisConfig,
    requested_stages: Sequence[str],
    resolved_stages: Sequence[str],
    resolved_pet_glob: str,
    pet_files: Sequence[Path],
    overwrite: bool,
    written_partitions: Sequence[Path],
    reused_partitions: Sequence[Path],
    artifacts: AnalysisArtifacts,
) -> None:
    """Write a portable record of background inputs, status, and products."""
    if not config.output.write_manifest:
        return

    manifest_path = background_manifest_path(config)
    manifest_data = {
        "schema_version": 1,
        "requested_stages": list(requested_stages),
        "resolved_stages": list(resolved_stages),
        "stage_status": artifacts.stage_status,
        "warnings": artifacts.warnings,
        "pet_glob": resolved_pet_glob,
        "pet_files": [str(path) for path in pet_files],
        "overwrite": overwrite,
        "written_partitions": [str(path) for path in written_partitions],
        "reused_partitions": [str(path) for path in reused_partitions],
        "dataset_paths": [str(path) for path in artifacts.dataset_paths],
        "yaml_paths": [str(path) for path in artifacts.yaml_paths],
        "plot_paths": [str(path) for path in artifacts.plot_paths],
    }
    write_yaml(manifest_data, manifest_path)
    artifacts.manifest_path = manifest_path
    logger.info("Background pipeline manifest written to %s", manifest_path)


def run_background_analysis(
    config: AnalysisConfig,
    *,
    stages: Sequence[str] = BACKGROUND_DEFAULT_STAGES,
    overwrite: bool | None = None,
) -> AnalysisArtifacts:
    """Build or reuse the run-aware background Parquet dataset."""
    artifacts = AnalysisArtifacts()
    requested_stages = tuple(stages)
    resolved_stages = resolve_background_stages(requested_stages)
    do_overwrite = overwrite if overwrite is not None else config.output.overwrite
    prod = config.production
    base = prod.reference_root / prod.version
    config_json_path = base / "config.json"

    resolved_pet_glob = config.background.pet_glob

    if not resolved_pet_glob:
        try:
            prod_cfg = dbetto.Props.read_from(str(config_json_path), subst_pathvar=True)
            resolved_pet_glob = (
                f"{prod_cfg['setups']['l200']['paths']['tier_pet']}/phy/*.lh5"
            )
        except Exception as e:
            logger.error(f"Could not resolve tier_pet from {config_json_path}: {e}")
            msg = "No pet_glob provided in yaml and could not read config.json"
            raise ValueError(msg) from e

    pet_files = discover_background_inputs(resolved_pet_glob)
    dataset_path = background_dataset_path(config)
    written_partitions: tuple[Path, ...] = ()
    reused_partitions: tuple[Path, ...] = ()

    if "build-dataset" in resolved_stages:
        has_cached_dataset = _has_parquet_fragments(dataset_path)
        if not pet_files and has_cached_dataset and not do_overwrite:
            artifacts.dataset_paths.append(dataset_path)
            artifacts.stage_status["build-dataset"] = "cached"
        elif not pet_files:
            _warn(
                artifacts,
                "Cannot build background dataset: no PET files match %s",
                resolved_pet_glob,
            )
            artifacts.stage_status["build-dataset"] = "blocked"
        else:
            context = build_analysis_context(config, load_eres=False)
            build_result = build_background_dataset(
                pet_files,
                dataset_path,
                context,
                apply_lar_veto=config.background.apply_lar_veto,
                comparison_cut_profile=config.background.comparison_cut_profile,
                overwrite=do_overwrite,
            )
            written_partitions = build_result.written_paths
            reused_partitions = build_result.reused_paths
            if written_partitions or reused_partitions:
                artifacts.dataset_paths.append(dataset_path)
            artifacts.stage_status["build-dataset"] = (
                "completed" if written_partitions else "cached"
            )

    # --- Check plots (automatic, not a separate stage) ----------------------
    if artifacts.dataset_paths and config.output.save_plots:
        plot_dir = config.paths.plots_root / "background"
        for energy_range in config.background.energy_ranges_keV:
            for bin_width in config.background.bin_widths_keV:
                plot_path = plot_background_spectrum(
                    dataset_path,
                    energy_range_keV=tuple(energy_range),
                    bin_width_keV=bin_width,
                    comparison_cut_profile=config.background.comparison_cut_profile,
                    output_dir=plot_dir,
                )
                artifacts.plot_paths.append(plot_path)

        summary_path, plot_warnings = plot_background_partition_summary(
            dataset_path, plot_dir
        )
        artifacts.plot_paths.append(summary_path)
        for warning in plot_warnings:
            _warn(artifacts, warning)

    _write_manifest(
        config,
        requested_stages,
        resolved_stages,
        resolved_pet_glob,
        pet_files,
        do_overwrite,
        written_partitions,
        reused_partitions,
        artifacts,
    )
    return artifacts

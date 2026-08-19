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
from bosonic_dm.pipeline.manifest import (
    current_background_cut_setup,
    current_background_plot_setup,
    load_background_manifest,
    manifest_stages,
    stage_record,
)
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
    """Return PET files matching a filename glob in stable order."""
    glob_path = Path(pet_glob)
    return tuple(
        sorted(path for path in glob_path.parent.glob(glob_path.name) if path.is_file())
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


def _parquet_fragments(dataset_path: Path) -> tuple[Path, ...]:
    """Return existing background Parquet fragments in stable order."""
    if not dataset_path.is_dir():
        return ()
    return tuple(sorted(dataset_path.rglob("*.parquet")))


def _expected_sanity_plot_paths(config: AnalysisConfig) -> list[Path]:
    """Return the deterministic background sanity-check plot paths."""
    plot_dir = config.paths.plots_root / "background"
    paths = [
        plot_dir / f"spectrum_{emin}_{emax}keV_{bin_width}keV.png"
        for emin, emax in config.background.energy_ranges_keV
        for bin_width in config.background.bin_widths_keV
    ]
    paths.append(plot_dir / "partition_summary.png")
    return paths


def _warn(artifacts: AnalysisArtifacts, message: str, *args: object) -> None:
    """Log and retain a non-fatal pipeline warning."""
    rendered = message % args if args else message
    logger.warning(rendered)
    artifacts.warnings.append(rendered)


def _write_manifest(
    config: AnalysisConfig,
    manifest: dict[str, object],
    artifacts: AnalysisArtifacts,
) -> None:
    """Atomically write the cumulative background manifest."""
    if not config.output.write_manifest:
        return

    manifest_path = background_manifest_path(config)
    write_yaml(manifest, manifest_path)
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
    manifest = load_background_manifest(background_manifest_path(config))
    manifest["data_root"] = str(config.paths.data_root)
    stages_manifest = manifest_stages(manifest)
    cut_setup = current_background_cut_setup(config)
    plot_setup = current_background_plot_setup(config)
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
    dataset_record = stage_record(manifest, "build-dataset")
    has_cached_dataset = _has_parquet_fragments(dataset_path)
    dataset_setup_matches = (
        dataset_record is not None and dataset_record.get("cut_setup") == cut_setup
    )
    dataset_recomputed = False
    dataset_is_current = False

    if "build-dataset" in resolved_stages:
        if has_cached_dataset and dataset_setup_matches and not do_overwrite:
            reused_partitions = _parquet_fragments(dataset_path)
            artifacts.dataset_paths.append(dataset_path)
            artifacts.stage_status["build-dataset"] = "cached"
            dataset_is_current = True
            stages_manifest["build-dataset"] = {
                "status": "cached",
                "outputs": [str(dataset_path)],
                "cut_setup": cut_setup,
                "pet_glob": resolved_pet_glob,
                "pet_files": [str(path) for path in pet_files],
                "written_partitions": [],
                "reused_partitions": [str(path) for path in reused_partitions],
            }
        elif not pet_files:
            _warn(
                artifacts,
                "Cannot build background dataset: no PET files match %s",
                resolved_pet_glob,
            )
            artifacts.stage_status["build-dataset"] = "blocked"
            blocked_record = dataset_record or {
                "outputs": [str(dataset_path)] if has_cached_dataset else []
            }
            stages_manifest["build-dataset"] = {
                **blocked_record,
                "status": "blocked",
                "pet_glob": resolved_pet_glob,
                "pet_files": [],
                "written_partitions": [],
                "reused_partitions": [],
            }
        else:
            context = build_analysis_context(config, load_eres=False)
            force_rebuild = do_overwrite or (
                has_cached_dataset and not dataset_setup_matches
            )
            build_result = build_background_dataset(
                pet_files,
                dataset_path,
                context,
                apply_lar_veto=config.apply_lar_veto,
                comparison_cut_profile=config.background.comparison_cut_profile,
                overwrite=force_rebuild,
            )
            written_partitions = build_result.written_paths
            reused_partitions = build_result.reused_paths
            if written_partitions or reused_partitions:
                artifacts.dataset_paths.append(dataset_path)
                dataset_is_current = True
            dataset_recomputed = bool(written_partitions)
            if written_partitions:
                dataset_status = "completed"
            elif reused_partitions:
                dataset_status = "cached"
            else:
                dataset_status = "blocked"
            artifacts.stage_status["build-dataset"] = dataset_status
            stages_manifest["build-dataset"] = {
                "status": dataset_status,
                "outputs": [str(dataset_path)] if dataset_is_current else [],
                "cut_setup": cut_setup,
                "pet_glob": resolved_pet_glob,
                "pet_files": [str(path) for path in pet_files],
                "written_partitions": [str(path) for path in written_partitions],
                "reused_partitions": [str(path) for path in reused_partitions],
            }

    # Sanity plots are automatic and deliberately not user-selectable.
    sanity_record = stage_record(manifest, "sanity-plots")
    expected_plot_paths = _expected_sanity_plot_paths(config)
    recorded_plot_paths = []
    if sanity_record is not None:
        outputs = sanity_record.get("outputs", [])
        if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
            recorded_plot_paths = [Path(path) for path in outputs]
    existing_plot_paths = list(
        dict.fromkeys(
            path
            for path in (*recorded_plot_paths, *expected_plot_paths)
            if path.exists()
        )
    )
    plot_setup_matches = (
        sanity_record is not None and sanity_record.get("plot_setup") == plot_setup
    )
    dataset_invalidated_plots = dataset_recomputed or (
        has_cached_dataset and not dataset_setup_matches
    )

    if config.output.save_plots and dataset_is_current:
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
        stages_manifest["sanity-plots"] = {
            "enabled": True,
            "status": "completed",
            "outputs": [str(path) for path in artifacts.plot_paths],
            "plot_setup": plot_setup,
        }
    elif config.output.save_plots:
        blocked_sanity_record = sanity_record or {
            "outputs": [str(path) for path in existing_plot_paths]
        }
        stages_manifest["sanity-plots"] = {
            **blocked_sanity_record,
            "enabled": True,
            "status": "blocked",
        }
    elif not existing_plot_paths:
        stages_manifest["sanity-plots"] = {
            "enabled": False,
            "status": "disabled",
            "outputs": [],
            "plot_setup": plot_setup,
        }
    elif dataset_invalidated_plots or not plot_setup_matches:
        stages_manifest["sanity-plots"] = {
            "enabled": False,
            "status": "stale",
            "outputs": [str(path) for path in existing_plot_paths],
            "plot_setup": (sanity_record.get("plot_setup") if sanity_record else None),
        }
    else:
        recorded_status = sanity_record.get("status") if sanity_record else None
        preserved_status = (
            recorded_status
            if recorded_status in {"completed", "cached", "partial"}
            else "stale"
        )
        stages_manifest["sanity-plots"] = {
            **(sanity_record or {}),
            "enabled": False,
            "status": preserved_status,
            "outputs": [str(path) for path in existing_plot_paths],
            "plot_setup": plot_setup,
        }

    manifest["last_run"] = {
        "requested_stages": list(requested_stages),
        "resolved_stages": list(resolved_stages),
        "overwrite": do_overwrite,
        "warnings": list(artifacts.warnings),
    }
    _write_manifest(config, manifest, artifacts)
    return artifacts

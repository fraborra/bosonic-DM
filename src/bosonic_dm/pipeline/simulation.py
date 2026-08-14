# Copyright (C) 2025 Francesco Borra
#

"""Simulation-analysis pipeline orchestration."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from dbetto import Props
from tqdm.auto import tqdm

from bosonic_dm.config import AnalysisConfig
from bosonic_dm.efficiency import build_labels_dicts, compute_efficiency_from_lazyframe
from bosonic_dm.io import build_parquet_dataset
from bosonic_dm.models import AnalysisArtifacts
from bosonic_dm.pipeline.context import build_analysis_context
from bosonic_dm.plotting.efficiency import (
    plot_efficiency_comparison,
    plot_fep_survival_fraction,
)
from bosonic_dm.plotting.spectra import (
    plot_aoe_survival_fraction,
    plot_lar_survival_fraction,
    plot_sim_energy_spectra,
)
from bosonic_dm.yaml_io import write_yaml

logger = logging.getLogger(__name__)

SIMULATION_STAGE_ORDER = (
    "count-vertices",
    "build-dataset",
    "efficiencies",
    "plots",
)
SIMULATION_DEFAULT_STAGES = SIMULATION_STAGE_ORDER
SIMULATION_STAGE_DEPENDENCIES = {
    "count-vertices": (),
    "build-dataset": (),
    "efficiencies": ("count-vertices", "build-dataset"),
    "plots": ("efficiencies",),
}


def resolve_simulation_stages(stages: Sequence[str]) -> tuple[str, ...]:
    """Expand requested stages with their dependencies in execution order."""
    unknown = set(stages) - set(SIMULATION_STAGE_ORDER)
    if unknown:
        msg = f"Unknown simulation stages: {sorted(unknown)}"
        raise ValueError(msg)

    resolved: set[str] = set()

    def add_with_dependencies(stage: str) -> None:
        for dependency in SIMULATION_STAGE_DEPENDENCIES[stage]:
            add_with_dependencies(dependency)
        resolved.add(stage)

    for requested_stage in stages:
        add_with_dependencies(requested_stage)

    return tuple(stage for stage in SIMULATION_STAGE_ORDER if stage in resolved)


def _warn(
    artifacts: AnalysisArtifacts,
    message: str,
    *args: object,
) -> None:
    """Log and retain a non-fatal pipeline warning."""
    rendered = message % args if args else message
    logger.warning(rendered)
    artifacts.warnings.append(rendered)


def _dataset_partition_path(dataset_dir: Path, energy: int) -> Path:
    """Return the deterministic Parquet path for one energy partition."""
    return dataset_dir / f"sim_e={energy}" / "data.parquet"


def _write_manifest(
    config: AnalysisConfig,
    interaction: str,
    requested_stages: Sequence[str],
    resolved_stages: Sequence[str],
    artifacts: AnalysisArtifacts,
) -> None:
    """Write a human-readable summary of pipeline products and skipped work."""
    if not config.output.write_manifest:
        return

    manifest_path = config.paths.data_root / f"{interaction}_manifest.yaml"
    manifest_data = {
        "interaction": interaction,
        "requested_stages": list(requested_stages),
        "resolved_stages": list(resolved_stages),
        "stage_status": artifacts.stage_status,
        "warnings": artifacts.warnings,
        "dataset_paths": [str(path) for path in artifacts.dataset_paths],
        "yaml_paths": [str(path) for path in artifacts.yaml_paths],
        "plot_paths": [str(path) for path in artifacts.plot_paths],
    }
    write_yaml(manifest_data, manifest_path)
    artifacts.manifest_path = manifest_path
    logger.info("Pipeline manifest written to %s", manifest_path)


def run_simulation_analysis(
    config: AnalysisConfig,
    interaction: str,
    *,
    stages: Sequence[str] = SIMULATION_DEFAULT_STAGES,
    overwrite: bool | None = None,
) -> AnalysisArtifacts:
    """Run simulation stages, skipping unavailable inputs without fabricating results."""
    artifacts = AnalysisArtifacts()

    if interaction not in config.interactions:
        msg = f"Unknown interaction: {interaction}"
        raise ValueError(msg)

    requested_stages = tuple(stages)
    resolved_stages = resolve_simulation_stages(requested_stages)
    int_cfg = config.interactions[interaction]
    dataset_name = int_cfg.name
    do_overwrite = overwrite if overwrite is not None else config.output.overwrite

    cvt_files: dict[int, list[Path]] = {}
    stp_files: dict[int, list[Path]] = {}
    for energy in config.energies_keV:
        job_string = int_cfg.job_template.format(energy=energy)
        cvt_dir = config.paths.simulation_root / "generated" / "tier" / "cvt"
        cvt_matches = sorted(cvt_dir.glob(f"l200cfg01-{job_string}-tier_cvt.lh5"))
        if cvt_matches:
            cvt_files[energy] = cvt_matches

        stp_dir = (
            config.paths.simulation_root / "generated" / "tier" / "stp" / job_string
        )
        stp_matches = sorted(stp_dir.glob(f"l200cfg01-{job_string}-job_*-tier_stp.lh5"))
        if stp_matches:
            stp_files[energy] = stp_matches

    counts_yaml_path = (
        config.paths.dictionaries_root / f"{interaction}_primary-counts.yaml"
    )
    vertex_counts: dict = {}
    if counts_yaml_path.exists() and not do_overwrite:
        vertex_counts = Props.read_from(str(counts_yaml_path))

    if "count-vertices" in resolved_stages:
        missing_count_energies = [
            energy
            for energy in config.energies_keV
            if do_overwrite or energy not in vertex_counts
        ]
        if not missing_count_energies:
            artifacts.stage_status["count-vertices"] = "cached"
            artifacts.yaml_paths.append(counts_yaml_path)
        else:
            first_energy = config.energies_keV[0]
            first_job = int_cfg.job_template.format(energy=first_energy)
            gdml_path = (
                config.paths.simulation_root
                / "generated"
                / "pars"
                / "geom"
                / f"l200cfg01-{first_job}-tier_stp-geom.gdml"
            )

            if not gdml_path.exists():
                _warn(
                    artifacts,
                    "Skipping vertex counting: GDML not found at %s",
                    gdml_path,
                )
                artifacts.stage_status["count-vertices"] = (
                    "partial" if vertex_counts else "blocked"
                )
                if vertex_counts:
                    artifacts.yaml_paths.append(counts_yaml_path)
            else:
                # Geometry dependencies are imported only when this stage runs.
                from bosonic_dm.geometry import (  # noqa: PLC0415
                    aggregate_vertex_counts,
                    assign_detectors_to_vertices,
                )

                if do_overwrite:
                    vertex_counts = {}

                for energy in tqdm(missing_count_energies, desc="Energies", position=0):
                    files = stp_files.get(energy, [])
                    if not files:
                        _warn(
                            artifacts,
                            "Skipping primary counts for %d keV: no STP files",
                            energy,
                        )
                        continue

                    detector_arrays, evtid_arrays = assign_detectors_to_vertices(
                        gdml=gdml_path,
                        lh5_files=files,
                        vtx_group="vtx",
                        save=False,
                        return_evtids=True,
                    )
                    if isinstance(detector_arrays, np.ndarray):
                        detector_arrays = [detector_arrays]
                        evtid_arrays = [evtid_arrays]
                    vertex_counts[energy] = aggregate_vertex_counts(
                        detector_arrays, evtid_arrays
                    )

                if vertex_counts:
                    write_yaml(vertex_counts, counts_yaml_path)
                    artifacts.yaml_paths.append(counts_yaml_path)

                still_missing = set(config.energies_keV) - set(vertex_counts)
                if not vertex_counts:
                    artifacts.stage_status["count-vertices"] = "blocked"
                elif still_missing:
                    artifacts.stage_status["count-vertices"] = "partial"
                else:
                    artifacts.stage_status["count-vertices"] = "completed"

    dataset_dir = config.paths.parquet_root / dataset_name
    if "build-dataset" in resolved_stages:
        energies_to_build = [
            energy
            for energy in config.energies_keV
            if do_overwrite or not _dataset_partition_path(dataset_dir, energy).exists()
        ]
        available_to_build = [
            energy for energy in energies_to_build if energy in cvt_files
        ]
        for energy in sorted(set(energies_to_build) - set(available_to_build)):
            _warn(
                artifacts,
                "Skipping dataset partition for %d keV: no CVT files",
                energy,
            )

        if available_to_build:
            build_parquet_dataset(
                energies=available_to_build,
                cvt_files=cvt_files,
                output_dir=dataset_dir,
                overwrite=do_overwrite,
            )

        available_partitions = [
            energy
            for energy in config.energies_keV
            if _dataset_partition_path(dataset_dir, energy).exists()
        ]
        if available_partitions:
            artifacts.dataset_paths.append(dataset_dir)

        if not available_partitions:
            artifacts.stage_status["build-dataset"] = "blocked"
        elif len(available_partitions) < len(config.energies_keV):
            artifacts.stage_status["build-dataset"] = "partial"
        elif not energies_to_build:
            artifacts.stage_status["build-dataset"] = "cached"
        else:
            artifacts.stage_status["build-dataset"] = "completed"

    if "efficiencies" in resolved_stages:
        if counts_yaml_path.exists() and not vertex_counts:
            vertex_counts = Props.read_from(str(counts_yaml_path))

        ready_energies = [
            energy
            for energy in config.energies_keV
            if _dataset_partition_path(dataset_dir, energy).exists()
            and energy in vertex_counts
        ]
        skipped_energies = set(config.energies_keV) - set(ready_energies)
        for energy in sorted(skipped_energies):
            _warn(
                artifacts,
                "Skipping efficiency for %d keV: dataset or primary counts missing",
                energy,
            )

        calibration_path = (
            config.paths.inputs_root / "dictionaries" / "eres_per_det_tot.yaml"
        )
        production_config = (
            config.production.reference_root / config.production.version / "config.json"
        )

        if not ready_energies:
            artifacts.stage_status["efficiencies"] = "blocked"
        elif not calibration_path.exists():
            _warn(
                artifacts,
                "Skipping efficiencies: calibration input not found at %s",
                calibration_path,
            )
            artifacts.stage_status["efficiencies"] = "blocked"
        elif not production_config.exists():
            _warn(
                artifacts,
                "Skipping efficiencies: production config not found at %s",
                production_config,
            )
            artifacts.stage_status["efficiencies"] = "blocked"
        else:
            context = build_analysis_context(config)
            chmap = context.get_channelmap_simulation()
            lf = pl.scan_parquet(dataset_dir / "*/*.parquet")
            efficiency_dict = compute_efficiency_from_lazyframe(
                lf=lf,
                eres_dict=context.eres_dict,
                simulated_energies=ready_energies,
                chmap=chmap,
                vertex_counts=vertex_counts,
                half_width_fwhm=config.fep_window.half_width_fwhm,
                selections=config.selections,
            )

            efficiency_path = (
                config.paths.dictionaries_root / f"{interaction}_efficiency.yaml"
            )
            write_yaml(efficiency_dict, efficiency_path)
            artifacts.yaml_paths.append(efficiency_path)
            artifacts.stage_status["efficiencies"] = (
                "partial" if skipped_energies else "completed"
            )

    if "plots" in resolved_stages:
        efficiency_path = (
            config.paths.dictionaries_root / f"{interaction}_efficiency.yaml"
        )
        if not efficiency_path.exists():
            _warn(
                artifacts,
                "Skipping plots for %s: efficiency dictionary not found",
                interaction,
            )
            artifacts.stage_status["plots"] = "blocked"
        else:
            efficiency_dict = Props.read_from(str(efficiency_path))

            # Avoid rebuilding context if already built in efficiencies stage
            if "context" not in locals():
                context = build_analysis_context(config)

            det_groups = Props.read_from(str(config.detector_groups))

            plot_configs = [
                ("detector_type", None, "png"),
                ("detector_group", det_groups, "png"),
            ]

            cfg = config.interactions[interaction]
            total_plots = len(plot_configs) * 3
            if cfg.make_energy_spectra_plots:
                total_plots += 1
            if cfg.make_lar_survival_plots:
                total_plots += 1
            if cfg.make_aoe_survival_plots:
                total_plots += 1

            pbar = tqdm(total=total_plots, desc=f"Plots ({interaction})")

            for group_by, groups, ext in plot_configs:
                labels_dicts = build_labels_dicts(
                    efficiency_dict,
                    eres_dict=context.eres_dict,
                    group_by=group_by,
                    detector_groups=groups,
                )

                for plot_type, title, ylabel, sharey in [
                    ("efficiency", "Efficiency", "Efficiency", True),
                    (
                        "eff_exp",
                        "Effective Exposure",
                        "Effective Exposure [kg yr]",
                        False,
                    ),
                ]:
                    save_path = None
                    if config.output.save_plots:
                        save_path = (
                            config.paths.plots_root
                            / f"{interaction}_{plot_type}_by_{group_by}.{ext}"
                        )

                    fig, _ = plot_efficiency_comparison(
                        labels_dicts=labels_dicts,
                        interaction=interaction,
                        plot_type=plot_type,
                        plot_title=title,
                        ylabel=ylabel,
                        sharey=sharey,
                        group_by=group_by,
                        save_path=save_path,
                    )

                    if save_path:
                        artifacts.plot_paths.append(save_path)

                    plt.close(fig)
                    pbar.update(1)

                # Extra plot for FEP survival fraction
                save_path_sf = None
                if config.output.save_plots:
                    save_path_sf = (
                        config.paths.plots_root
                        / f"{interaction}_fep_survival_fraction_by_{group_by}.{ext}"
                    )

                fig_sf, _ = plot_fep_survival_fraction(
                    labels_dicts=labels_dicts,
                    interaction=interaction,
                    plot_title="FEP Survival Fraction",
                    group_by=group_by,
                    save_path=save_path_sf,
                )

                if save_path_sf:
                    artifacts.plot_paths.append(save_path_sf)

                plt.close(fig_sf)
                pbar.update(1)

            artifacts.stage_status["plots"] = "completed"

            # Extra simulation plots
            if (
                cfg.make_lar_survival_plots
                or cfg.make_energy_spectra_plots
                or cfg.make_aoe_survival_plots
            ):
                parquet_dir = config.paths.parquet_root / interaction
                if parquet_dir.exists():
                    lf = pl.scan_parquet(str(parquet_dir), hive_partitioning=True)
                    plot_save_dir = (
                        config.paths.plots_root / interaction
                        if config.output.save_plots
                        else Path("plots")
                    )

                    if cfg.make_energy_spectra_plots:
                        rawid_path = (
                            config.paths.inputs_root
                            / "dictionaries/rawid_by_det_type.yaml"
                        )
                        if rawid_path.exists():
                            rawid_by_det_type = Props.read_from(str(rawid_path))
                            plot_sim_energy_spectra(
                                lf=lf,
                                simulated_energies=config.energies_keV,
                                rawid_by_det_type=rawid_by_det_type,
                                interaction=interaction,
                                save_dir=plot_save_dir / "energy_spectra",
                            )
                        else:
                            _warn(
                                artifacts,
                                "Cannot plot sim energy spectra: %s missing",
                                rawid_path,
                            )
                        pbar.update(1)

                    if cfg.make_lar_survival_plots or cfg.make_aoe_survival_plots:
                        chmap = context.get_channelmap_simulation()

                    if cfg.make_lar_survival_plots:
                        plot_lar_survival_fraction(
                            lf=lf,
                            simulated_energies=config.energies_keV,
                            chmap=chmap,
                            interaction=interaction,
                            save_dir=plot_save_dir / "lar_survival",
                        )
                        pbar.update(1)

                    if cfg.make_aoe_survival_plots:
                        plot_aoe_survival_fraction(
                            lf=lf,
                            simulated_energies=config.energies_keV,
                            chmap=chmap,
                            interaction=interaction,
                            save_dir=plot_save_dir / "aoe_survival",
                        )
                        pbar.update(1)
                else:
                    _warn(
                        artifacts,
                        "Parquet dir %s missing. Cannot generate advanced plots.",
                        parquet_dir,
                    )

                    skipped = 0
                    if cfg.make_energy_spectra_plots:
                        skipped += 1
                    if cfg.make_lar_survival_plots:
                        skipped += 1
                    if cfg.make_aoe_survival_plots:
                        skipped += 1
                    pbar.update(skipped)

            pbar.close()

    _write_manifest(
        config,
        interaction,
        requested_stages,
        resolved_stages,
        artifacts,
    )
    return artifacts

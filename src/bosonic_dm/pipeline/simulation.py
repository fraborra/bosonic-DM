from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
from dbetto import Props

from bosonic_dm.config import AnalysisConfig
from bosonic_dm.efficiency import compute_efficiency_from_lazyframe
from bosonic_dm.geometry import aggregate_vertex_counts, assign_detectors_to_vertices
from bosonic_dm.io import build_parquet_dataset
from bosonic_dm.models import AnalysisArtifacts
from bosonic_dm.pipeline.context import build_analysis_context
from bosonic_dm.yaml_io import write_yaml

logger = logging.getLogger(__name__)


def run_simulation_analysis(
    config: AnalysisConfig,
    interaction: str,
    *,
    stages: Sequence[str] = (
        "count-vertices",
        "build-dataset",
        "efficiencies",
        "plots",
    ),
    overwrite: bool | None = None,
) -> AnalysisArtifacts:
    """Run the simulation analysis pipeline for a specific interaction."""
    artifacts = AnalysisArtifacts(
        dataset_paths=[],
        yaml_paths=[],
        plot_paths=[],
    )

    if interaction not in config.interactions:
        msg = f"Unknown interaction: {interaction}"
        raise ValueError(msg)

    int_cfg = config.interactions[interaction]
    dataset_name = int_cfg.name

    # Resolve overwrite flag
    do_overwrite = overwrite if overwrite is not None else config.output.overwrite

    # Step 1: Create shared context
    context = build_analysis_context(config)
    chmap = context.get_channelmap_simulation()

    # Discover and validate LH5 inputs
    # For CVT
    cvt_files: dict[int, list[Path]] = {}
    stp_files: dict[int, list[Path]] = {}

    for ene in config.energies_keV:
        job_string = int_cfg.job_template.format(energy=ene)

        # CVT
        search_dir_cvt = config.paths.simulation_root / "generated" / "tier" / "cvt"
        filename_pattern_cvt = f"l200cfg01-{job_string}-tier_cvt.lh5"
        cvt_matches = sorted(search_dir_cvt.glob(filename_pattern_cvt))
        if cvt_matches:
            cvt_files[ene] = cvt_matches

        # STP
        search_dir_stp = (
            config.paths.simulation_root / "generated" / "tier" / "stp" / job_string
        )
        filename_pattern_stp = f"l200cfg01-{job_string}-job_*-tier_stp.lh5"
        stp_matches = sorted(search_dir_stp.glob(filename_pattern_stp))
        if stp_matches:
            stp_files[ene] = stp_matches

    # Step 2: "Pre-efficiency" vertex counts
    counts_yaml_path = (
        config.paths.dictionaries_root / f"{interaction}_primary-counts.yaml"
    )

    if "count-vertices" in stages:
        if not counts_yaml_path.exists() or do_overwrite:
            logger.info("Computing vertex counts from STP files per energy...")
            vertex_counts_per_ene = {}
            first_ene = config.energies_keV[0]
            first_job_string = int_cfg.job_template.format(energy=first_ene)
            gdml_path = (
                config.paths.simulation_root
                / "generated"
                / "pars"
                / "geom"
                / f"l200cfg01-{first_job_string}-tier_stp-geom.gdml"
            )
            if hasattr(config.paths, "gdml") and config.paths.gdml is not None:
                gdml_path = config.paths.gdml

            if gdml_path.exists():
                for ene, files in stp_files.items():
                    det_name_arrays = assign_detectors_to_vertices(
                        gdml=gdml_path, lh5_files=files, vtx_group="vtx", save=False
                    )

                    if isinstance(det_name_arrays, np.ndarray):
                        det_name_arrays = [det_name_arrays]

                    vertex_counts_per_ene[ene] = aggregate_vertex_counts(
                        det_name_arrays
                    )

                write_yaml(vertex_counts_per_ene, counts_yaml_path)
            else:
                logger.warning(
                    "GDML not found at %s. Cannot compute vertex counts.", gdml_path
                )
        else:
            logger.info(
                "Vertex counts already exist at %s, skipping.", counts_yaml_path
            )

    # Load vertex counts if available
    vertex_counts = {}
    if counts_yaml_path.exists():
        vertex_counts = Props.read_from(str(counts_yaml_path))

    # Step 3: Build Parquet cache
    if "build-dataset" in stages:
        logger.info("Building parquet dataset...")
        build_parquet_dataset(
            energies=config.energies_keV,
            cvt_files=cvt_files,
            output_dir=config.paths.parquet_root / dataset_name,
            overwrite=do_overwrite,
        )
        artifacts.dataset_paths.append(config.paths.parquet_root / dataset_name)

    # Step 4: Compute efficiencies
    if "efficiencies" in stages:
        logger.info("Computing efficiencies...")
        lf = pl.scan_parquet(config.paths.parquet_root / dataset_name / "*/*.parquet")

        eff_dict = compute_efficiency_from_lazyframe(
            lf=lf,
            eres_dict=context.eres_dict,
            simulated_energies=config.energies_keV,
            chmap=chmap,
            vertex_counts=vertex_counts,
            half_width_fwhm=config.fep_window.half_width_fwhm,
        )

        eff_path = config.paths.dictionaries_root / f"{interaction}_efficiency.yaml"
        write_yaml(eff_dict, eff_path)
        artifacts.yaml_paths.append(eff_path)

    # Step 5: Plots
    if "plots" in stages:
        pass  # To be implemented in notebook or future steps

    if config.output.write_manifest:
        manifest_path = config.paths.data_root / f"{interaction}_manifest.yaml"
        manifest_data = {
            "dataset_paths": [str(p) for p in artifacts.dataset_paths],
            "yaml_paths": [str(p) for p in artifacts.yaml_paths],
            "plot_paths": [str(p) for p in artifacts.plot_paths],
        }
        write_yaml(manifest_data, manifest_path)
        artifacts = replace(artifacts, manifest_path=manifest_path)
        logger.info("Pipeline manifest written to %s", manifest_path)

    return artifacts

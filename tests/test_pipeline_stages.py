# Copyright (C) 2025 Francesco Borra
#

from __future__ import annotations

from pathlib import Path

import pytest

from bosonic_dm.config import (
    AnalysisConfig,
    BackgroundConfig,
    FepWindowConfig,
    InteractionConfig,
    OutputConfig,
    PathsConfig,
    ProductionConfig,
)
from bosonic_dm.pipeline.simulation import (
    current_cut_setup,
    resolve_simulation_stages,
    run_simulation_analysis,
)
from bosonic_dm.yaml_io import read_yaml


def test_efficiency_stage_adds_its_dependencies() -> None:
    assert resolve_simulation_stages(["efficiencies"]) == (
        "count-vertices",
        "build-dataset",
        "efficiencies",
    )


def test_plot_stage_adds_transitive_dependencies() -> None:
    assert resolve_simulation_stages(["plots"]) == (
        "count-vertices",
        "build-dataset",
        "efficiencies",
        "plots",
    )


def test_unknown_stage_is_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_simulation_stages(["unknown"])


def test_missing_inputs_soft_block_dependent_stages(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    config = AnalysisConfig(
        production=ProductionConfig(
            version="v1",
            reference_root=tmp_path / "production",
            metadata_override=None,
        ),
        paths=PathsConfig(
            simulation_root=tmp_path / "simulation",
            data_root=data_root,
            parquet_root=data_root / "parquet",
            dictionaries_root=data_root / "dictionaries",
            inputs_root=tmp_path / "calibration",
            plots_root=tmp_path / "plots",
            temporary_root=tmp_path / "tmp",
        ),
        energies_keV=(200,),
        fep_window=FepWindowConfig(half_width_fwhm=2.0),
        selections=("all",),
        apply_lar_veto=True,
        interactions={
            "axio-electric": InteractionConfig(
                name="axio-electric",
                job_template="electron_{energy}keV_hpge_bulk",
                make_lar_survival_plots=False,
                make_energy_spectra_plots=False,
                make_aoe_survival_plots=False,
            )
        },
        background=BackgroundConfig(
            pet_glob="",
            comparison_cut_profile="without-bb-like",
            energy_ranges_keV=[(20, 300)],
            bin_widths_keV=[5],
        ),
        output=OutputConfig(
            overwrite=False,
            save_plots=True,
            write_manifest=True,
        ),
        detector_groups=tmp_path / "groups.yaml",
    )

    artifacts = run_simulation_analysis(
        config,
        interaction="axio-electric",
        stages=("efficiencies",),
    )

    assert artifacts.stage_status == {
        "count-vertices": "blocked",
        "build-dataset": "blocked",
        "efficiencies": "blocked",
    }
    assert artifacts.warnings
    manifest = read_yaml(data_root / "axio-electric_manifest.yaml")
    assert manifest["schema_version"] == 2
    assert manifest["stages"]["efficiencies"]["status"] == "blocked"
    assert manifest["last_run"]["resolved_stages"] == [
        "count-vertices",
        "build-dataset",
        "efficiencies",
    ]
    cut_setup = current_cut_setup(config)
    assert cut_setup["efficiency_output_schema_version"] == 2
    assert cut_setup["selection_metadata"] == {
        "energy_window": {
            "reconstructed_energy_column": "energy",
            "center_energy_column": "sim_e",
            "half_width_fwhm": 2.0,
            "resolution_scope": "period-run-detector",
        },
        "good_channel": {
            "column": "is_good_channel",
            "pass_value": True,
        },
        "lar_veto": {
            "applied": True,
            "column": "coincident_spms",
            "pass_value": False,
            "null_policy": "reject",
        },
        "selections": {
            "all": {
                "event_requirements": {},
            }
        },
    }

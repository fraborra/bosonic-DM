# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bosonic_dm.background import BackgroundDatasetBuildResult
from bosonic_dm.config import (
    AnalysisConfig,
    BackgroundConfig,
    FepWindowConfig,
    OutputConfig,
    PathsConfig,
    ProductionConfig,
)
from bosonic_dm.pipeline.background import (
    BACKGROUND_DEFAULT_STAGES,
    background_dataset_path,
    resolve_background_stages,
    run_background_analysis,
)
from bosonic_dm.yaml_io import read_yaml


def _make_config(
    tmp_path: Path,
    *,
    write_manifest: bool = True,
) -> AnalysisConfig:
    data_root = tmp_path / "data"
    return AnalysisConfig(
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
            inputs_root=tmp_path / "inputs",
            plots_root=tmp_path / "plots",
            temporary_root=tmp_path / "tmp",
        ),
        energies_keV=(200,),
        fep_window=FepWindowConfig(half_width_fwhm=1.0),
        selections=(),
        interactions={},
        background=BackgroundConfig(
            pet_glob=str(tmp_path / "pet" / "*.lh5"),
            apply_lar_veto=True,
            comparison_cut_profile="without-bb-like",
            energy_ranges_keV=[(20, 300)],
            bin_widths_keV=[5],
        ),
        output=OutputConfig(
            overwrite=False,
            save_plots=False,
            write_manifest=write_manifest,
        ),
        detector_groups=tmp_path / "groups.yaml",
    )


def test_background_default_stage_is_build_dataset() -> None:
    assert BACKGROUND_DEFAULT_STAGES == ("build-dataset",)
    assert resolve_background_stages(BACKGROUND_DEFAULT_STAGES) == (
        "build-dataset",
    )


def test_unknown_background_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown background stages"):
        resolve_background_stages(("summaries",))


def test_missing_pet_inputs_block_dataset_and_write_manifest(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    artifacts = run_background_analysis(config)

    assert artifacts.stage_status == {"build-dataset": "blocked"}
    assert artifacts.dataset_paths == []
    assert artifacts.warnings == [
        f"Cannot build background dataset: no PET files match {config.background.pet_glob}"
    ]
    assert artifacts.manifest_path == tmp_path / "data/background_manifest.yaml"

    manifest = read_yaml(artifacts.manifest_path)
    assert manifest["requested_stages"] == ["build-dataset"]
    assert manifest["resolved_stages"] == ["build-dataset"]
    assert manifest["stage_status"] == {"build-dataset": "blocked"}
    assert manifest["pet_files"] == []


def test_cached_dataset_does_not_require_pet_inputs(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    dataset_path = background_dataset_path(config)
    fragment = dataset_path / "period=p03/run=r000/data.parquet"
    fragment.parent.mkdir(parents=True)
    fragment.write_bytes(b"cached")

    artifacts = run_background_analysis(config)

    assert artifacts.stage_status == {"build-dataset": "cached"}
    assert artifacts.dataset_paths == [dataset_path]
    assert artifacts.warnings == []


def test_overwrite_requires_pet_inputs_even_with_cached_dataset(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    dataset_path = background_dataset_path(config)
    fragment = dataset_path / "period=p03/run=r000/data.parquet"
    fragment.parent.mkdir(parents=True)
    fragment.write_bytes(b"cached")

    artifacts = run_background_analysis(config, overwrite=True)

    assert artifacts.stage_status == {"build-dataset": "blocked"}
    assert artifacts.dataset_paths == []
    assert artifacts.warnings


@patch("bosonic_dm.pipeline.background.build_background_dataset")
@patch("bosonic_dm.pipeline.background.build_analysis_context")
def test_discovered_pet_inputs_are_built_and_recorded(
    mock_context: MagicMock,
    mock_build: MagicMock,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    pet_path = tmp_path / "pet/l200-p03-r000-phy-tier_pet.lh5"
    pet_path.parent.mkdir(parents=True)
    pet_path.write_bytes(b"placeholder")
    fragment = (
        background_dataset_path(config)
        / "period=p03/run=r000/l200-p03-r000-phy-tier_pet.parquet"
    )
    mock_build.return_value = BackgroundDatasetBuildResult(
        written_paths=(fragment,),
        reused_paths=(),
    )

    artifacts = run_background_analysis(config)

    assert artifacts.stage_status == {"build-dataset": "completed"}
    assert artifacts.dataset_paths == [background_dataset_path(config)]
    assert artifacts.warnings == []
    mock_context.assert_called_once_with(config, load_eres=False)
    mock_build.assert_called_once()
    manifest = read_yaml(artifacts.manifest_path)
    assert manifest["pet_files"] == [str(pet_path)]
    assert manifest["written_partitions"] == [str(fragment)]


def test_manifest_can_be_disabled(tmp_path: Path) -> None:
    config = _make_config(tmp_path, write_manifest=False)

    artifacts = run_background_analysis(config)

    assert artifacts.manifest_path is None
    assert not (tmp_path / "data/background_manifest.yaml").exists()


@patch("bosonic_dm.pipeline.background.plot_background_partition_summary")
@patch("bosonic_dm.pipeline.background.plot_background_spectrum")
@patch("bosonic_dm.pipeline.background.build_background_dataset")
@patch("bosonic_dm.pipeline.background.build_analysis_context")
def test_save_plots_populates_plot_paths_and_manifest(
    mock_context: MagicMock,
    mock_build: MagicMock,
    mock_spectrum: MagicMock,
    mock_summary: MagicMock,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    # Enable save_plots
    config = AnalysisConfig(
        production=config.production,
        paths=config.paths,
        energies_keV=config.energies_keV,
        fep_window=config.fep_window,
        selections=config.selections,
        interactions=config.interactions,
        background=config.background,
        output=OutputConfig(
            overwrite=False, save_plots=True, write_manifest=True
        ),
        detector_groups=config.detector_groups,
    )
    pet_path = tmp_path / "pet/l200-p03-r000-phy-tier_pet.lh5"
    pet_path.parent.mkdir(parents=True, exist_ok=True)
    pet_path.write_bytes(b"placeholder")

    fragment = (
        background_dataset_path(config)
        / "period=p03/run=r000/l200-p03-r000-phy-tier_pet.parquet"
    )
    mock_build.return_value = BackgroundDatasetBuildResult(
        written_paths=(fragment,),
        reused_paths=(),
    )
    spectrum_path = tmp_path / "plots/background/spectrum_20_300keV_5keV.png"
    summary_path = tmp_path / "plots/background/partition_summary.png"
    mock_spectrum.return_value = spectrum_path
    mock_summary.return_value = (summary_path, [])

    artifacts = run_background_analysis(config)

    assert artifacts.stage_status["build-dataset"] == "completed"
    # One energy_range × one bin_width = 1 spectrum + 1 summary = 2 plots
    assert len(artifacts.plot_paths) == 2
    assert spectrum_path in artifacts.plot_paths
    assert summary_path in artifacts.plot_paths
    mock_spectrum.assert_called_once()
    mock_summary.assert_called_once()

    manifest = read_yaml(artifacts.manifest_path)
    assert len(manifest["plot_paths"]) == 2


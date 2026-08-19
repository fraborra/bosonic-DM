# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

from dataclasses import replace
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
    current_background_cut_setup,
    current_background_plot_setup,
    resolve_background_stages,
    run_background_analysis,
)
from bosonic_dm.yaml_io import read_yaml, write_yaml


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
        apply_lar_veto=True,
        interactions={},
        background=BackgroundConfig(
            pet_glob=str(tmp_path / "pet" / "*.lh5"),
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


def _write_matching_manifest(config: AnalysisConfig) -> None:
    dataset_path = background_dataset_path(config)
    write_yaml(
        {
            "schema_version": 2,
            "data_root": str(config.paths.data_root),
            "stages": {
                "build-dataset": {
                    "status": "completed",
                    "outputs": [str(dataset_path)],
                    "cut_setup": current_background_cut_setup(config),
                }
            },
        },
        config.paths.data_root / "background_manifest.yaml",
    )


def _add_sanity_record(
    config: AnalysisConfig,
    plot_paths: list[Path],
    *,
    status: str = "completed",
) -> None:
    manifest_path = config.paths.data_root / "background_manifest.yaml"
    manifest = read_yaml(manifest_path)
    manifest["stages"]["sanity-plots"] = {
        "enabled": True,
        "status": status,
        "outputs": [str(path) for path in plot_paths],
        "plot_setup": current_background_plot_setup(config),
    }
    write_yaml(manifest, manifest_path)


def test_background_default_stage_is_build_dataset() -> None:
    assert BACKGROUND_DEFAULT_STAGES == ("build-dataset",)
    assert resolve_background_stages(BACKGROUND_DEFAULT_STAGES) == ("build-dataset",)


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
    assert manifest["last_run"]["requested_stages"] == ["build-dataset"]
    assert manifest["last_run"]["resolved_stages"] == ["build-dataset"]
    assert manifest["stages"]["build-dataset"]["status"] == "blocked"
    assert manifest["stages"]["build-dataset"]["pet_files"] == []
    assert manifest["stages"]["sanity-plots"]["status"] == "disabled"


def test_cached_dataset_does_not_require_pet_inputs(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    dataset_path = background_dataset_path(config)
    fragment = dataset_path / "period=p03/run=r000/data.parquet"
    fragment.parent.mkdir(parents=True)
    fragment.write_bytes(b"cached")
    _write_matching_manifest(config)

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
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_bytes(b"dataset")
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
    dataset_stage = manifest["stages"]["build-dataset"]
    assert dataset_stage["pet_files"] == [str(pet_path)]
    assert dataset_stage["written_partitions"] == [str(fragment)]

    second_artifacts = run_background_analysis(config)
    assert second_artifacts.stage_status == {"build-dataset": "cached"}
    mock_build.assert_called_once()


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
    mock_context: MagicMock,  # noqa: ARG001
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
        apply_lar_veto=config.apply_lar_veto,
        interactions=config.interactions,
        background=config.background,
        output=OutputConfig(overwrite=False, save_plots=True, write_manifest=True),
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
    # One energy_range x one bin_width = 1 spectrum + 1 summary = 2 plots
    assert len(artifacts.plot_paths) == 2
    assert spectrum_path in artifacts.plot_paths
    assert summary_path in artifacts.plot_paths
    mock_spectrum.assert_called_once()
    mock_summary.assert_called_once()

    manifest = read_yaml(artifacts.manifest_path)
    sanity_stage = manifest["stages"]["sanity-plots"]
    assert sanity_stage["enabled"] is True
    assert sanity_stage["status"] == "completed"
    assert len(sanity_stage["outputs"]) == 2


@patch("bosonic_dm.pipeline.background.build_background_dataset")
@patch("bosonic_dm.pipeline.background.build_analysis_context")
def test_changed_background_cut_setup_forces_dataset_rebuild(
    mock_context: MagicMock,  # noqa: ARG001
    mock_build: MagicMock,
    tmp_path: Path,
) -> None:
    old_config = _make_config(tmp_path)
    config = replace(old_config, apply_lar_veto=False)
    dataset_path = background_dataset_path(config)
    fragment = dataset_path / "period=p03/run=r000/data.parquet"
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_bytes(b"old")
    pet_path = tmp_path / "pet/l200-p03-r000-phy-tier_pet.lh5"
    pet_path.parent.mkdir(parents=True, exist_ok=True)
    pet_path.write_bytes(b"input")
    _write_matching_manifest(old_config)
    mock_build.return_value = BackgroundDatasetBuildResult(
        written_paths=(fragment,),
        reused_paths=(),
    )

    artifacts = run_background_analysis(config)

    assert artifacts.stage_status["build-dataset"] == "completed"
    assert mock_build.call_args.kwargs["overwrite"] is True
    manifest = read_yaml(artifacts.manifest_path)
    assert manifest["stages"]["build-dataset"][
        "cut_setup"
    ] == current_background_cut_setup(config)


def test_disabling_plots_preserves_valid_sanity_status(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    dataset_path = background_dataset_path(config)
    fragment = dataset_path / "period=p03/run=r000/data.parquet"
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_bytes(b"dataset")
    plot_paths = [
        config.paths.plots_root / "background/spectrum_20_300keV_5keV.png",
        config.paths.plots_root / "background/partition_summary.png",
    ]
    for plot_path in plot_paths:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path.write_bytes(b"plot")
    _write_matching_manifest(config)
    _add_sanity_record(config, plot_paths)

    artifacts = run_background_analysis(config)

    manifest = read_yaml(artifacts.manifest_path)
    sanity_stage = manifest["stages"]["sanity-plots"]
    assert sanity_stage["enabled"] is False
    assert sanity_stage["status"] == "completed"
    assert sanity_stage["outputs"] == [str(path) for path in plot_paths]


@patch("bosonic_dm.pipeline.background.build_background_dataset")
@patch("bosonic_dm.pipeline.background.build_analysis_context")
def test_rebuilding_dataset_with_plots_disabled_marks_sanity_plots_stale(
    mock_context: MagicMock,  # noqa: ARG001
    mock_build: MagicMock,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    dataset_path = background_dataset_path(config)
    fragment = dataset_path / "period=p03/run=r000/data.parquet"
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_bytes(b"dataset")
    pet_path = tmp_path / "pet/l200-p03-r000-phy-tier_pet.lh5"
    pet_path.parent.mkdir(parents=True, exist_ok=True)
    pet_path.write_bytes(b"input")
    plot_paths = [
        config.paths.plots_root / "background/spectrum_20_300keV_5keV.png",
        config.paths.plots_root / "background/partition_summary.png",
    ]
    for plot_path in plot_paths:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path.write_bytes(b"plot")
    _write_matching_manifest(config)
    _add_sanity_record(config, plot_paths)
    mock_build.return_value = BackgroundDatasetBuildResult(
        written_paths=(fragment,),
        reused_paths=(),
    )

    artifacts = run_background_analysis(config, overwrite=True)

    manifest = read_yaml(artifacts.manifest_path)
    sanity_stage = manifest["stages"]["sanity-plots"]
    assert sanity_stage["enabled"] is False
    assert sanity_stage["status"] == "stale"
    assert all(path.exists() for path in plot_paths)


@patch("bosonic_dm.pipeline.background.plot_background_partition_summary")
@patch("bosonic_dm.pipeline.background.plot_background_spectrum")
def test_reenabling_plots_regenerates_stale_sanity_outputs(
    mock_spectrum: MagicMock,
    mock_summary: MagicMock,
    tmp_path: Path,
) -> None:
    disabled_config = _make_config(tmp_path)
    config = replace(
        disabled_config,
        output=replace(disabled_config.output, save_plots=True),
    )
    dataset_path = background_dataset_path(config)
    fragment = dataset_path / "period=p03/run=r000/data.parquet"
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_bytes(b"dataset")
    plot_paths = [
        config.paths.plots_root / "background/spectrum_20_300keV_5keV.png",
        config.paths.plots_root / "background/partition_summary.png",
    ]
    for plot_path in plot_paths:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path.write_bytes(b"old plot")
    _write_matching_manifest(config)
    _add_sanity_record(config, plot_paths, status="stale")
    mock_spectrum.return_value = plot_paths[0]
    mock_summary.return_value = (plot_paths[1], [])

    artifacts = run_background_analysis(config)

    mock_spectrum.assert_called_once()
    mock_summary.assert_called_once()
    manifest = read_yaml(artifacts.manifest_path)
    sanity_stage = manifest["stages"]["sanity-plots"]
    assert sanity_stage["enabled"] is True
    assert sanity_stage["status"] == "completed"

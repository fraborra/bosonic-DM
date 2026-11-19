# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import dbetto
import numpy as np
import polars as pl
import pytest

from bosonic_dm.background import BackgroundDatasetBuildResult, spectrum_in_range
from bosonic_dm.config import (
    AnalysisConfig,
    BackgroundConfig,
    FepWindowConfig,
    OutputConfig,
    PathsConfig,
    ProductionConfig,
)
from bosonic_dm.fitting.background_fit import rate_prior_centres
from bosonic_dm.pipeline.background import (
    BACKGROUND_DEFAULT_STAGES,
    background_dataset_path,
    current_background_cut_setup,
    current_background_plot_setup,
    resolve_background_groups,
    resolve_background_stages,
    run_background_analysis,
)
from bosonic_dm.pipeline.manifest import current_background_fit_setup
from bosonic_dm.yaml_io import read_yaml, write_yaml

DATASET_ONLY = ("build-dataset",)


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


def test_background_default_stages_are_all_stages() -> None:
    assert BACKGROUND_DEFAULT_STAGES == ("build-dataset", "fit", "fit-plots")
    assert resolve_background_stages(DATASET_ONLY) == ("build-dataset",)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (("fit-plots",), ("build-dataset", "fit", "fit-plots")),
        (("fit",), ("build-dataset", "fit")),
        (("fit", "build-dataset"), ("build-dataset", "fit")),
    ],
)
def test_background_stage_dependencies_are_resolved(
    requested: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    assert resolve_background_stages(requested) == expected


def test_unknown_background_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown background stages"):
        resolve_background_stages(("summaries",))


def test_missing_pet_inputs_block_dataset_and_write_manifest(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    artifacts = run_background_analysis(config, stages=DATASET_ONLY)

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

    artifacts = run_background_analysis(config, stages=DATASET_ONLY)

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

    artifacts = run_background_analysis(config, stages=DATASET_ONLY, overwrite=True)

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

    artifacts = run_background_analysis(config, stages=DATASET_ONLY)

    assert artifacts.stage_status == {"build-dataset": "completed"}
    assert artifacts.dataset_paths == [background_dataset_path(config)]
    assert artifacts.warnings == []
    mock_context.assert_called_once_with(config, load_eres=False)
    mock_build.assert_called_once()
    manifest = read_yaml(artifacts.manifest_path)
    dataset_stage = manifest["stages"]["build-dataset"]
    assert dataset_stage["pet_files"] == [str(pet_path)]
    assert dataset_stage["written_partitions"] == [str(fragment)]

    second_artifacts = run_background_analysis(config, stages=DATASET_ONLY)
    assert second_artifacts.stage_status == {"build-dataset": "cached"}
    mock_build.assert_called_once()


def test_manifest_can_be_disabled(tmp_path: Path) -> None:
    config = _make_config(tmp_path, write_manifest=False)

    artifacts = run_background_analysis(config, stages=DATASET_ONLY)

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

    artifacts = run_background_analysis(config, stages=DATASET_ONLY)

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

    artifacts = run_background_analysis(config, stages=DATASET_ONLY)

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

    artifacts = run_background_analysis(config, stages=DATASET_ONLY)

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

    artifacts = run_background_analysis(config, stages=DATASET_ONLY, overwrite=True)

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

    artifacts = run_background_analysis(config, stages=DATASET_ONLY)

    mock_spectrum.assert_called_once()
    mock_summary.assert_called_once()
    manifest = read_yaml(artifacts.manifest_path)
    sanity_stage = manifest["stages"]["sanity-plots"]
    assert sanity_stage["enabled"] is True
    assert sanity_stage["status"] == "completed"


# --- fit and fit-plots stages ----------------------------------------------

GROUPS = {
    "ICPC group1": {"types": "icpc"},
    "BEGe": {"types": "bege"},
}
PREFIT = {
    "N_2nu": 1000.0,
    "N_ar39": 5000.0,
    "linear_parameters": [0.1, 0.1],
    "ar39_parameters": [1.0, 1.0, 1.0, 1.0, 1.0],
}


def _synthetic_spectrum() -> dbetto.AttrsDict:
    edges = np.arange(60.0, 2001.0, 1.0)
    n_bins = len(edges) - 1
    energy = np.array([150.0, 400.0, 800.0])
    counts = np.full(n_bins, 3)
    return dbetto.AttrsDict(
        {
            "energy": energy,
            "bin_edges": edges,
            "expo": 1.0,
            "bin_centres": 0.5 * (edges[:-1] + edges[1:]),
            "bin_widths_keV": np.diff(edges),
            "fit_mask": np.ones(n_bins, dtype=bool),
            "counts_analysis": counts,
            "counts_comparison": counts,
            "rate_analysis": counts.astype(float),
            "rate_comparison": counts.astype(float),
            "mask_analysis": np.ones(len(energy), dtype=bool),
            "mask_comparison": np.ones(len(energy), dtype=bool),
        }
    )


def _make_fit_config(tmp_path: Path, *, save_plots: bool = False) -> AnalysisConfig:
    config = _make_config(tmp_path)
    config = replace(config, output=replace(config.output, save_plots=save_plots))
    write_yaml(GROUPS, config.detector_groups)
    dictionaries = config.paths.inputs_root / "dictionaries"
    write_yaml({}, dictionaries / "eres_dict.yaml")
    write_yaml(
        {"ICPC_group1": {"fit_ar": PREFIT}, "BEGe": {"fit_ar": PREFIT}},
        dictionaries / "prefit_dictionary.yaml",
    )
    fragment = background_dataset_path(config) / "period=p03/run=r000/data.parquet"
    fragment.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"energy": [1.0]}).write_parquet(fragment)
    _write_matching_manifest(config)
    return config


def _fake_save(
    fit: object,  # noqa: ARG001
    summary: object,  # noqa: ARG001
    data: object,  # noqa: ARG001
    output_dir: Path,
    **kwargs: object,  # noqa: ARG001
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "posterior": output_dir / "posterior.nc",
        "bhat": output_dir / "bhat.parquet",
        "draws": output_dir / "expected_counts_draws.npy",
        "summary": output_dir / "summary.yaml",
    }
    for path in paths.values():
        path.write_bytes(b"fit")
    return paths


def _fake_plot(
    *args: object,  # noqa: ARG001
    output_path: Path,
    **kwargs: object,  # noqa: ARG001
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"plot")


@pytest.fixture
def fit_mocks() -> Iterator[dict[str, MagicMock]]:
    module = "bosonic_dm.pipeline.background"
    with (
        patch(f"{module}.prepare_background_spectra") as prepare,
        patch(f"{module}.compute_group_exposure", return_value=2.0),
        patch(f"{module}.CmdStanModel") as model,
        patch(f"{module}.run_background_fit") as run_fit,
        patch(f"{module}.summarize_background_fit") as summarize,
        patch(f"{module}.save_background_fit", side_effect=_fake_save) as save,
        patch(f"{module}.load_background_fit") as load_fit,
        patch(f"{module}.open_background_posterior") as open_posterior,
        patch(f"{module}.plot_background_fit_residuals", side_effect=_fake_plot) as res,
        patch(f"{module}.plot_background_chains", side_effect=_fake_plot) as chains,
        patch(f"{module}.plot_background_corner", side_effect=_fake_plot) as corner,
    ):
        prepare.side_effect = lambda _lf, groups, *_a, **_k: {
            group: _synthetic_spectrum() for group in groups
        }
        summarize.return_value = MagicMock(passed=True)
        yield {
            "prepare": prepare,
            "model": model,
            "run_fit": run_fit,
            "save": save,
            "load_fit": load_fit,
            "open_posterior": open_posterior,
            "residuals": res,
            "chains": chains,
            "corner": corner,
        }


def test_requested_groups_accept_underscores() -> None:
    assert resolve_background_groups(["ICPC_group1"], GROUPS) == ("ICPC group1",)
    assert resolve_background_groups(None, GROUPS) == ("ICPC group1", "BEGe")


def test_unknown_group_lists_valid_groups(tmp_path: Path) -> None:
    config = _make_fit_config(tmp_path)

    with pytest.raises(ValueError, match="Valid groups: \\['ICPC group1', 'BEGe'\\]"):
        run_background_analysis(config, stages=("fit",), groups=["COAX"])


def test_fit_runs_per_group_and_records_manifest(
    tmp_path: Path, fit_mocks: dict[str, MagicMock]
) -> None:
    config = _make_fit_config(tmp_path)

    artifacts = run_background_analysis(config, stages=("fit",))

    assert artifacts.stage_status == {"build-dataset": "cached", "fit": "completed"}
    assert fit_mocks["run_fit"].call_count == 2
    fit_mocks["model"].assert_called_once()
    run_kwargs = fit_mocks["run_fit"].call_args.kwargs
    assert run_kwargs["seed"] == 12345
    assert run_kwargs["chains"] == 4
    data = fit_mocks["run_fit"].call_args.args[1]
    assert data["fit_c0"] == 0
    # Fit range [100, 1100] keV with 1 keV bins.
    assert data["n_bins"] == 1000
    assert fit_mocks["save"].call_args.args[3] == (
        config.paths.data_root / "background_fits" / "BEGe"
    )

    manifest = read_yaml(artifacts.manifest_path)
    record = manifest["stages"]["fit"]["groups"]["ICPC group1"]
    assert record["fit_setup"] == current_background_fit_setup(config)
    assert (
        record["dataset_fingerprint"]
        == (manifest["stages"]["build-dataset"]["dataset_fingerprint"])
    )
    assert all(
        Path(path).parent == config.paths.data_root / "background_fits/ICPC_group1"
        for path in record["outputs"]
    )


def test_rate_priors_use_external_centres(
    tmp_path: Path, fit_mocks: dict[str, MagicMock]
) -> None:
    config = _make_fit_config(tmp_path)

    run_background_analysis(config, stages=("fit",), groups=["BEGe"])

    data = fit_mocks["run_fit"].call_args.args[1]
    n2nu, ar39 = rate_prior_centres(
        spectrum_in_range(_synthetic_spectrum(), (100.0, 1100.0)), 2.0
    )
    assert n2nu == pytest.approx(2.4e3 * 2.0 * 0.90 / 0.88 * 0.9)
    assert ar39 == pytest.approx(3 * 465 / 465 * 565)
    np.testing.assert_allclose(data["log_rate_prior_mean"][:2], np.log([n2nu, ar39]))
    np.testing.assert_allclose(data["log_rate_prior_sd"][:2], [0.5, 2.0])


def test_current_fit_is_reused_and_changed_setup_refits(
    tmp_path: Path, fit_mocks: dict[str, MagicMock]
) -> None:
    config = _make_fit_config(tmp_path)
    run_background_analysis(config, stages=("fit",), groups=["BEGe"])
    assert fit_mocks["run_fit"].call_count == 1

    artifacts = run_background_analysis(config, stages=("fit",), groups=["BEGe"])
    assert artifacts.stage_status["fit"] == "cached"
    assert fit_mocks["run_fit"].call_count == 1

    background = config.background
    changed = replace(
        config,
        background=replace(background, fit=replace(background.fit, shape_sd=0.7)),
    )
    artifacts = run_background_analysis(changed, stages=("fit",), groups=["BEGe"])
    assert artifacts.stage_status["fit"] == "completed"
    assert fit_mocks["run_fit"].call_count == 2


@patch("bosonic_dm.pipeline.background.build_background_dataset")
@patch("bosonic_dm.pipeline.background.build_analysis_context")
def test_overwrite_forces_refit(
    mock_context: MagicMock,  # noqa: ARG001
    mock_build: MagicMock,
    tmp_path: Path,
    fit_mocks: dict[str, MagicMock],
) -> None:
    config = _make_fit_config(tmp_path)
    run_background_analysis(config, stages=("fit",), groups=["BEGe"])
    assert fit_mocks["run_fit"].call_count == 1
    pet_path = tmp_path / "pet/l200-p03-r000-phy-tier_pet.lh5"
    pet_path.parent.mkdir(parents=True, exist_ok=True)
    pet_path.write_bytes(b"input")
    fragment = background_dataset_path(config) / "period=p03/run=r000/data.parquet"
    mock_build.return_value = BackgroundDatasetBuildResult(
        written_paths=(fragment,), reused_paths=()
    )

    artifacts = run_background_analysis(
        config, stages=("fit",), groups=["BEGe"], overwrite=True
    )

    assert artifacts.stage_status["fit"] == "completed"
    assert fit_mocks["run_fit"].call_count == 2


@patch("bosonic_dm.pipeline.background.build_background_dataset")
@patch("bosonic_dm.pipeline.background.build_analysis_context")
def test_rebuilt_dataset_marks_fit_records_stale(
    mock_context: MagicMock,  # noqa: ARG001
    mock_build: MagicMock,
    tmp_path: Path,
    fit_mocks: dict[str, MagicMock],
) -> None:
    config = _make_fit_config(tmp_path)
    run_background_analysis(config, stages=("fit",), groups=["BEGe"])
    pet_path = tmp_path / "pet/l200-p03-r000-phy-tier_pet.lh5"
    pet_path.parent.mkdir(parents=True, exist_ok=True)
    pet_path.write_bytes(b"input")
    fragment = background_dataset_path(config) / "period=p03/run=r000/data.parquet"
    mock_build.return_value = BackgroundDatasetBuildResult(
        written_paths=(fragment,), reused_paths=()
    )

    artifacts = run_background_analysis(config, stages=DATASET_ONLY, overwrite=True)

    manifest = read_yaml(artifacts.manifest_path)
    assert manifest["stages"]["fit"]["status"] == "stale"
    assert manifest["stages"]["fit"]["groups"]["BEGe"]["status"] == "stale"

    run_background_analysis(config, stages=("fit",), groups=["BEGe"])
    assert fit_mocks["run_fit"].call_count == 2


def test_fit_plots_are_drawn_from_saved_fit_and_reused(
    tmp_path: Path, fit_mocks: dict[str, MagicMock]
) -> None:
    config = _make_fit_config(tmp_path, save_plots=True)

    with (
        patch("bosonic_dm.pipeline.background.plot_background_spectrum"),
        patch(
            "bosonic_dm.pipeline.background.plot_background_partition_summary",
            return_value=(tmp_path / "summary.png", []),
        ),
    ):
        artifacts = run_background_analysis(
            config, stages=("fit-plots",), groups=["ICPC_group1"]
        )
        assert artifacts.stage_status["fit-plots"] == "completed"
        plot_dir = config.paths.plots_root / "background_fit" / "ICPC_group1"
        expected = [
            plot_dir / f"{name}.png" for name in ("residuals", "chains", "corner")
        ]
        assert all(path in artifacts.plot_paths for path in expected)
        fit_mocks["load_fit"].assert_called_once_with(
            config.paths.data_root / "background_fits" / "ICPC_group1"
        )
        posterior = fit_mocks["open_posterior"].return_value
        assert fit_mocks["chains"].call_args.args[0] is posterior
        manifest = read_yaml(artifacts.manifest_path)
        record = manifest["stages"]["fit-plots"]["groups"]["ICPC group1"]
        assert record["outputs"] == [str(path) for path in expected]

        artifacts = run_background_analysis(
            config, stages=("fit-plots",), groups=["ICPC_group1"]
        )
    assert artifacts.stage_status["fit-plots"] == "cached"
    assert fit_mocks["residuals"].call_count == 1
    assert fit_mocks["run_fit"].call_count == 1


def test_fit_plots_honour_save_plots(
    tmp_path: Path, fit_mocks: dict[str, MagicMock]
) -> None:
    config = _make_fit_config(tmp_path, save_plots=False)

    artifacts = run_background_analysis(config, groups=["BEGe"])

    assert artifacts.stage_status["fit-plots"] == "disabled"
    fit_mocks["residuals"].assert_not_called()


def test_fit_is_blocked_without_dataset(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    write_yaml(GROUPS, config.detector_groups)

    artifacts = run_background_analysis(config, stages=("fit",))

    assert artifacts.stage_status["fit"] == "blocked"

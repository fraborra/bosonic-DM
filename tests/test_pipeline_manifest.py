# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

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
from bosonic_dm.pipeline.manifest import current_cut_setup, current_plot_setup
from bosonic_dm.pipeline.simulation import run_simulation_analysis
from bosonic_dm.yaml_io import read_yaml, write_yaml

INTERACTION = "axio-electric"


def _make_config(tmp_path: Path) -> AnalysisConfig:
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
        fep_window=FepWindowConfig(half_width_fwhm=2.0),
        selections=("all", "valid-psd"),
        apply_lar_veto=True,
        interactions={
            INTERACTION: InteractionConfig(
                name=INTERACTION,
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


def _simulation_paths(config: AnalysisConfig) -> tuple[Path, Path, Path, Path]:
    counts_path = config.paths.dictionaries_root / f"{INTERACTION}_primary-counts.yaml"
    dataset_path = config.paths.parquet_root / INTERACTION
    partition_path = dataset_path / "sim_e=200/data.parquet"
    efficiency_path = config.paths.dictionaries_root / f"{INTERACTION}_efficiency.yaml"
    return counts_path, dataset_path, partition_path, efficiency_path


def _materialize_cached_inputs(config: AnalysisConfig) -> None:
    counts_path, _, partition_path, efficiency_path = _simulation_paths(config)
    write_yaml({200: {"V00000A": 10}}, counts_path)
    partition_path.parent.mkdir(parents=True, exist_ok=True)
    partition_path.write_bytes(b"cached parquet")
    write_yaml({200: {}}, efficiency_path)


def _write_simulation_manifest(
    config: AnalysisConfig,
    *,
    efficiency_setup: dict[str, object] | None,
    include_plots: bool = False,
) -> None:
    counts_path, dataset_path, _, efficiency_path = _simulation_paths(config)
    efficiency_record: dict[str, object] = {
        "status": "completed",
        "outputs": [str(efficiency_path)],
    }
    if efficiency_setup is not None:
        efficiency_record["cut_setup"] = efficiency_setup
    stages: dict[str, object] = {
        "count-vertices": {
            "status": "completed",
            "outputs": [str(counts_path)],
        },
        "build-dataset": {
            "status": "completed",
            "outputs": [str(dataset_path)],
        },
        "efficiencies": efficiency_record,
    }
    if include_plots:
        plot_paths = _basic_plot_paths(config)
        stages["plots"] = {
            "status": "completed",
            "outputs": [str(path) for path in plot_paths],
            "cut_setup": current_cut_setup(config),
            "plot_setup": current_plot_setup(config, INTERACTION),
        }
    write_yaml(
        {
            "schema_version": 2,
            "interaction": INTERACTION,
            "data_root": str(config.paths.data_root),
            "stages": stages,
        },
        config.paths.data_root / f"{INTERACTION}_manifest.yaml",
    )


def _basic_plot_paths(config: AnalysisConfig) -> list[Path]:
    return [
        config.paths.plots_root / f"{INTERACTION}_{plot_type}_by_{group_by}.png"
        for group_by in ("detector_type", "detector_group")
        for plot_type in ("efficiency", "eff_exp", "fep_survival_fraction")
    ]


def test_later_command_preserves_prior_stage_records(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    run_simulation_analysis(config, INTERACTION, stages=("count-vertices",))
    first_manifest = read_yaml(config.paths.data_root / f"{INTERACTION}_manifest.yaml")

    run_simulation_analysis(config, INTERACTION, stages=("build-dataset",))
    second_manifest = read_yaml(config.paths.data_root / f"{INTERACTION}_manifest.yaml")

    assert (
        second_manifest["stages"]["count-vertices"]
        == first_manifest["stages"]["count-vertices"]
    )
    assert second_manifest["last_run"]["requested_stages"] == ["build-dataset"]


def test_interaction_manifests_remain_independent(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    dark_compton = InteractionConfig(
        name="dark-compton",
        job_template="bosonic_dark_compton_{energy}keV",
        make_lar_survival_plots=False,
        make_energy_spectra_plots=False,
        make_aoe_survival_plots=False,
    )
    config = replace(
        config,
        interactions={**config.interactions, "dark-compton": dark_compton},
    )

    run_simulation_analysis(config, INTERACTION, stages=("count-vertices",))
    run_simulation_analysis(config, "dark-compton", stages=("count-vertices",))

    axio_manifest = read_yaml(config.paths.data_root / f"{INTERACTION}_manifest.yaml")
    dark_manifest = read_yaml(config.paths.data_root / "dark-compton_manifest.yaml")
    assert axio_manifest["interaction"] == INTERACTION
    assert dark_manifest["interaction"] == "dark-compton"


def test_matching_setup_reuses_counts_dataset_and_efficiency(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _materialize_cached_inputs(config)
    _write_simulation_manifest(config, efficiency_setup=current_cut_setup(config))

    with patch("bosonic_dm.pipeline.simulation.build_analysis_context") as mock_context:
        artifacts = run_simulation_analysis(
            config,
            INTERACTION,
            stages=("efficiencies",),
        )

    assert artifacts.stage_status == {
        "count-vertices": "cached",
        "build-dataset": "cached",
        "efficiencies": "cached",
    }
    mock_context.assert_not_called()


def test_overwrite_recomputes_every_resolved_simulation_stage(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    _materialize_cached_inputs(config)
    _write_simulation_manifest(config, efficiency_setup=current_cut_setup(config))
    job_name = "electron_200keV_hpge_bulk"
    gdml_path = (
        config.paths.simulation_root
        / "generated/pars/geom"
        / f"l200cfg01-{job_name}-tier_stp-geom.gdml"
    )
    stp_path = (
        config.paths.simulation_root
        / "generated/tier/stp"
        / job_name
        / f"l200cfg01-{job_name}-job_000-tier_stp.lh5"
    )
    cvt_path = (
        config.paths.simulation_root
        / "generated/tier/cvt"
        / f"l200cfg01-{job_name}-tier_cvt.lh5"
    )
    for input_path in (gdml_path, stp_path, cvt_path):
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_bytes(b"input")
    calibration_path = config.paths.inputs_root / "dictionaries/eres_per_det_tot.yaml"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text("{}\n")
    production_path = (
        config.production.reference_root / config.production.version / "config.json"
    )
    production_path.parent.mkdir(parents=True, exist_ok=True)
    production_path.write_text("{}\n")
    _, _, partition_path, _ = _simulation_paths(config)

    def rebuild_dataset(**_kwargs: object) -> None:
        partition_path.write_bytes(b"rebuilt")

    with (
        patch(
            "bosonic_dm.pipeline.simulation.assign_detectors_to_vertices",
            return_value=([object()], [object()]),
        ) as mock_assign,
        patch(
            "bosonic_dm.pipeline.simulation.aggregate_vertex_counts",
            return_value={"V00000A": 20},
        ),
        patch(
            "bosonic_dm.pipeline.simulation.build_parquet_dataset",
            side_effect=rebuild_dataset,
        ) as mock_build,
        patch("bosonic_dm.pipeline.simulation.pl.scan_parquet"),
        patch(
            "bosonic_dm.pipeline.simulation.compute_efficiency_from_lazyframe",
            return_value={200: {}},
        ) as mock_compute,
        patch("bosonic_dm.pipeline.simulation.build_analysis_context") as mock_context,
    ):
        mock_context.return_value.eres_dict = {}
        mock_context.return_value.get_channelmap_simulation.return_value = object()
        artifacts = run_simulation_analysis(
            config,
            INTERACTION,
            stages=("efficiencies",),
            overwrite=True,
        )

    assert artifacts.stage_status == {
        "count-vertices": "completed",
        "build-dataset": "completed",
        "efficiencies": "completed",
    }
    mock_assign.assert_called_once()
    assert mock_build.call_args.kwargs["overwrite"] is True
    mock_compute.assert_called_once()


@pytest.mark.parametrize("change", ["missing", "fep", "selections", "lar"])
@patch("bosonic_dm.pipeline.simulation.compute_efficiency_from_lazyframe")
@patch("bosonic_dm.pipeline.simulation.pl.scan_parquet")
@patch("bosonic_dm.pipeline.simulation.build_analysis_context")
def test_missing_or_changed_cut_setup_recomputes_efficiency(
    mock_context: MagicMock,
    mock_scan: MagicMock,
    mock_compute: MagicMock,
    change: str,
    tmp_path: Path,
) -> None:
    original_config = _make_config(tmp_path)
    config = original_config
    if change == "fep":
        config = replace(config, fep_window=FepWindowConfig(half_width_fwhm=1.5))
    elif change == "selections":
        config = replace(config, selections=("all", "sse"))
    elif change == "lar":
        config = replace(config, apply_lar_veto=False)

    _materialize_cached_inputs(config)
    recorded_setup = None if change == "missing" else current_cut_setup(original_config)
    _write_simulation_manifest(config, efficiency_setup=recorded_setup)
    calibration_path = config.paths.inputs_root / "dictionaries/eres_per_det_tot.yaml"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text("{}\n")
    production_path = (
        config.production.reference_root / config.production.version / "config.json"
    )
    production_path.parent.mkdir(parents=True, exist_ok=True)
    production_path.write_text("{}\n")
    mock_context.return_value.eres_dict = {}
    mock_context.return_value.get_channelmap_simulation.return_value = object()
    mock_compute.return_value = {200: {}}

    artifacts = run_simulation_analysis(
        config,
        INTERACTION,
        stages=("efficiencies",),
    )

    assert artifacts.stage_status["efficiencies"] == "completed"
    mock_scan.assert_called_once()
    mock_compute.assert_called_once()
    manifest = read_yaml(config.paths.data_root / f"{INTERACTION}_manifest.yaml")
    assert manifest["stages"]["efficiencies"]["cut_setup"] == current_cut_setup(config)


@patch("bosonic_dm.pipeline.simulation.build_parquet_dataset")
def test_recomputed_dataset_marks_downstream_stages_stale(
    mock_build: MagicMock,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    _materialize_cached_inputs(config)
    _, _, partition_path, _ = _simulation_paths(config)
    partition_path.unlink()
    _write_simulation_manifest(
        config,
        efficiency_setup=current_cut_setup(config),
        include_plots=True,
    )
    cvt_path = (
        config.paths.simulation_root
        / "generated/tier/cvt"
        / "l200cfg01-electron_200keV_hpge_bulk-tier_cvt.lh5"
    )
    cvt_path.parent.mkdir(parents=True, exist_ok=True)
    cvt_path.write_bytes(b"input")

    def create_partition(**_kwargs: object) -> None:
        partition_path.parent.mkdir(parents=True, exist_ok=True)
        partition_path.write_bytes(b"rebuilt")

    mock_build.side_effect = create_partition
    run_simulation_analysis(config, INTERACTION, stages=("build-dataset",))

    manifest = read_yaml(config.paths.data_root / f"{INTERACTION}_manifest.yaml")
    assert manifest["stages"]["efficiencies"]["status"] == "stale"
    assert manifest["stages"]["plots"]["status"] == "stale"


def test_matching_plot_setup_and_outputs_reuses_plots(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _materialize_cached_inputs(config)
    for plot_path in _basic_plot_paths(config):
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path.write_bytes(b"plot")
    _write_simulation_manifest(
        config,
        efficiency_setup=current_cut_setup(config),
        include_plots=True,
    )

    with patch("bosonic_dm.pipeline.simulation.build_analysis_context") as mock_context:
        artifacts = run_simulation_analysis(config, INTERACTION, stages=("plots",))

    assert artifacts.stage_status["plots"] == "cached"
    assert artifacts.plot_paths == _basic_plot_paths(config)
    mock_context.assert_not_called()


def test_changed_plot_setup_regenerates_only_plots(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _materialize_cached_inputs(config)
    write_yaml({}, config.detector_groups)
    for plot_path in _basic_plot_paths(config):
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path.write_bytes(b"old plot")
    _write_simulation_manifest(
        config,
        efficiency_setup=current_cut_setup(config),
        include_plots=True,
    )
    manifest_path = config.paths.data_root / f"{INTERACTION}_manifest.yaml"
    manifest = read_yaml(manifest_path)
    manifest["stages"]["plots"]["plot_setup"]["detector_groups"] = "old.yaml"
    write_yaml(manifest, manifest_path)

    with (
        patch("bosonic_dm.pipeline.simulation.build_analysis_context") as mock_context,
        patch(
            "bosonic_dm.pipeline.simulation.build_labels_dicts",
            return_value={},
        ),
        patch(
            "bosonic_dm.pipeline.simulation.plot_efficiency_comparison",
            return_value=(object(), None),
        ) as mock_efficiency_plot,
        patch(
            "bosonic_dm.pipeline.simulation.plot_fep_survival_fraction",
            return_value=(object(), None),
        ) as mock_survival_plot,
        patch("bosonic_dm.pipeline.simulation.plt.close"),
    ):
        mock_context.return_value.eres_dict = {}
        artifacts = run_simulation_analysis(config, INTERACTION, stages=("plots",))

    assert artifacts.stage_status == {
        "count-vertices": "cached",
        "build-dataset": "cached",
        "efficiencies": "cached",
        "plots": "completed",
    }
    assert mock_efficiency_plot.call_count == 4
    assert mock_survival_plot.call_count == 2

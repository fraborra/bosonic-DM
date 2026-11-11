# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
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
from bosonic_dm.efficiency import efficiency_table, write_efficiency_table
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


# Primary counts stored in the cached efficiency table, and the ones the
# mocked geometry returns when primaries are counted again.
CACHED_COUNTS = {"V00000A": {"p03": {"r000": 10}}}
COUNTED = {"V00000A": {"p03": {"r000": 20}}}
SIMULATION = "bosonic_dm.pipeline.simulation"


def _simulation_paths(config: AnalysisConfig) -> tuple[Path, Path, Path]:
    dataset_path = config.paths.parquet_root / INTERACTION
    partition_path = dataset_path / "sim_e=200/data.parquet"
    efficiency_path = (
        config.paths.dictionaries_root / f"{INTERACTION}_efficiency.parquet"
    )
    return dataset_path, partition_path, efficiency_path


def _cached_efficiency_table() -> pl.DataFrame:
    """Efficiency table at 200 keV; only its primary counts matter here."""
    run = {
        "n_primaries": 10,
        "expo": 1.0,
        "selections": {"all": {"status": "valid"}},
    }
    return efficiency_table({200: {"V00000A": {"period_runs": {"p03": {"r000": run}}}}})


def _materialize_cached_inputs(config: AnalysisConfig) -> None:
    _, partition_path, efficiency_path = _simulation_paths(config)
    partition_path.parent.mkdir(parents=True, exist_ok=True)
    partition_path.write_bytes(b"cached parquet")
    write_efficiency_table(_cached_efficiency_table(), efficiency_path)


def _job_file(config: AnalysisConfig, tier: str, energy: int, job: str) -> Path:
    job_name = f"electron_{energy}keV_hpge_bulk"
    return (
        config.paths.simulation_root
        / "generated/tier"
        / tier
        / job_name
        / f"l200cfg01-{job_name}-job_{job}-tier_{tier}.lh5"
    )


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"input")
    return path


def _materialize_simulation_inputs(
    config: AnalysisConfig,
    energy: int = 200,
    *,
    stp_jobs: tuple[str, ...] = ("000",),
    evt_jobs: tuple[str, ...] = ("000",),
) -> list[Path]:
    """Create the GDML and the per-job STP and evt files; return the evt files."""
    first_job = f"electron_{config.energies_keV[0]}keV_hpge_bulk"
    _touch(
        config.paths.simulation_root
        / "generated/pars/geom"
        / f"l200cfg01-{first_job}-tier_stp-geom.gdml"
    )
    for job in stp_jobs:
        _touch(_job_file(config, "stp", energy, job))
    return [_touch(_job_file(config, "evt", energy, job)) for job in evt_jobs]


def _materialize_calibration_inputs(config: AnalysisConfig) -> None:
    calibration_path = config.paths.inputs_root / "dictionaries/eres_per_det_tot.yaml"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text("{}\n")
    production_path = (
        config.production.reference_root / config.production.version / "config.json"
    )
    production_path.parent.mkdir(parents=True, exist_ok=True)
    production_path.write_text("{}\n")


def _write_simulation_manifest(
    config: AnalysisConfig,
    *,
    efficiency_setup: dict[str, object] | None,
    include_plots: bool = False,
    legacy_count_record: bool = False,
) -> None:
    dataset_path, _, efficiency_path = _simulation_paths(config)
    efficiency_record: dict[str, object] = {
        "status": "completed",
        "outputs": [str(efficiency_path)],
    }
    if efficiency_setup is not None:
        efficiency_record["cut_setup"] = efficiency_setup
    stages: dict[str, object] = {
        "build-dataset": {
            "status": "completed",
            "outputs": [str(dataset_path)],
        },
        "efficiencies": efficiency_record,
    }
    if legacy_count_record:
        stages["count-vertices"] = {"status": "completed", "outputs": ["old.yaml"]}
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


def _read_manifest(config: AnalysisConfig) -> dict:
    return read_yaml(config.paths.data_root / f"{INTERACTION}_manifest.yaml")


def _basic_plot_paths(config: AnalysisConfig) -> list[Path]:
    return [
        config.paths.plots_root / f"{INTERACTION}_{plot_type}_by_{group_by}.png"
        for group_by in ("detector_type", "detector_group")
        for plot_type in ("efficiency", "eff_exp", "fep_survival_fraction")
    ]


@pytest.fixture
def mock_assign() -> Iterator[MagicMock]:
    """Replace the geometry and evt readers used to count primaries."""
    with (
        patch(
            f"{SIMULATION}.assign_detectors_to_vertices",
            return_value=([object()], [object()]),
        ) as assign,
        patch(f"{SIMULATION}.read_run_ranges", return_value=[("p03", "r000", 0, 99)]),
        patch(f"{SIMULATION}.aggregate_vertex_counts_by_run", return_value=COUNTED),
    ):
        yield assign


@pytest.fixture
def mock_compute() -> Iterator[MagicMock]:
    """Replace the metadata, the dataset scan and the efficiency computation."""
    with (
        patch(f"{SIMULATION}.pl.scan_parquet"),
        patch(f"{SIMULATION}.build_analysis_context") as context,
        patch(
            f"{SIMULATION}.compute_efficiency_from_lazyframe",
            return_value={200: {}},
        ) as compute,
    ):
        context.return_value.eres_dict = {}
        context.return_value.get_channelmap_simulation.return_value = object()
        yield compute


def test_later_command_preserves_prior_stage_records(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    run_simulation_analysis(config, INTERACTION, stages=("efficiencies",))
    first_manifest = _read_manifest(config)

    run_simulation_analysis(config, INTERACTION, stages=("build-dataset",))
    second_manifest = _read_manifest(config)

    assert (
        second_manifest["stages"]["efficiencies"]
        == first_manifest["stages"]["efficiencies"]
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

    run_simulation_analysis(config, INTERACTION, stages=("build-dataset",))
    run_simulation_analysis(config, "dark-compton", stages=("build-dataset",))

    axio_manifest = _read_manifest(config)
    dark_manifest = read_yaml(config.paths.data_root / "dark-compton_manifest.yaml")
    assert axio_manifest["interaction"] == INTERACTION
    assert dark_manifest["interaction"] == "dark-compton"


def test_matching_setup_reuses_dataset_and_efficiency(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _materialize_cached_inputs(config)
    _write_simulation_manifest(config, efficiency_setup=current_cut_setup(config))

    with patch(f"{SIMULATION}.build_analysis_context") as mock_context:
        artifacts = run_simulation_analysis(
            config,
            INTERACTION,
            stages=("efficiencies",),
        )

    assert artifacts.stage_status == {
        "build-dataset": "cached",
        "efficiencies": "cached",
    }
    mock_context.assert_not_called()


def test_overwrite_recomputes_every_resolved_simulation_stage(
    tmp_path: Path, mock_assign: MagicMock, mock_compute: MagicMock
) -> None:
    config = _make_config(tmp_path)
    _materialize_cached_inputs(config)
    _write_simulation_manifest(
        config,
        efficiency_setup=current_cut_setup(config),
        legacy_count_record=True,
    )
    evt_files = _materialize_simulation_inputs(config)
    _materialize_calibration_inputs(config)
    _, partition_path, _ = _simulation_paths(config)

    def rebuild_dataset(**_kwargs: object) -> None:
        partition_path.write_bytes(b"rebuilt")

    with patch(
        f"{SIMULATION}.build_parquet_dataset", side_effect=rebuild_dataset
    ) as mock_build:
        artifacts = run_simulation_analysis(
            config,
            INTERACTION,
            stages=("efficiencies",),
            overwrite=True,
        )

    assert artifacts.stage_status == {
        "build-dataset": "completed",
        "efficiencies": "completed",
    }
    # The dataset is built from the per-job evt files, not from a cvt file.
    assert mock_build.call_args.kwargs["overwrite"] is True
    assert mock_build.call_args.kwargs["evt_files"] == {200: evt_files}
    # Overwrite counts the primaries again instead of reusing the table.
    mock_assign.assert_called_once()
    mock_compute.assert_called_once()
    assert mock_compute.call_args.kwargs["vertex_counts"] == {200: COUNTED}

    manifest = _read_manifest(config)
    assert "count-vertices" not in manifest["stages"]
    assert manifest["stages"]["efficiencies"]["primary_counts"] == {
        "reused_energies_keV": [],
        "counted_energies_keV": [200],
    }
    # Primary counts are kept in the efficiency table only.
    assert sorted(path.name for path in config.paths.dictionaries_root.iterdir()) == [
        f"{INTERACTION}_efficiency.parquet"
    ]


@pytest.mark.parametrize("change", ["fep", "selections", "lar"])
def test_changed_cut_setup_recomputes_efficiency_with_the_stored_counts(
    change: str, tmp_path: Path, mock_assign: MagicMock, mock_compute: MagicMock
) -> None:
    original_config = _make_config(tmp_path)
    if change == "fep":
        config = replace(
            original_config, fep_window=FepWindowConfig(half_width_fwhm=1.5)
        )
    elif change == "selections":
        config = replace(original_config, selections=("all", "sse"))
    else:
        config = replace(original_config, apply_lar_veto=False)

    _materialize_cached_inputs(config)
    _write_simulation_manifest(
        config, efficiency_setup=current_cut_setup(original_config)
    )
    _materialize_calibration_inputs(config)

    artifacts = run_simulation_analysis(
        config,
        INTERACTION,
        stages=("efficiencies",),
    )

    assert artifacts.stage_status["efficiencies"] == "completed"
    mock_compute.assert_called_once()
    # The counts do not depend on the cuts: they come from the cached table.
    mock_assign.assert_not_called()
    assert mock_compute.call_args.kwargs["vertex_counts"] == {200: CACHED_COUNTS}
    record = _read_manifest(config)["stages"]["efficiencies"]
    assert record["cut_setup"] == current_cut_setup(config)
    assert record["primary_counts"] == {
        "reused_energies_keV": [200],
        "counted_energies_keV": [],
    }


def test_table_without_a_recorded_setup_gets_its_primaries_recounted(
    tmp_path: Path, mock_assign: MagicMock, mock_compute: MagicMock
) -> None:
    config = _make_config(tmp_path)
    _materialize_cached_inputs(config)
    _write_simulation_manifest(config, efficiency_setup=None)
    _materialize_simulation_inputs(config)
    _materialize_calibration_inputs(config)

    artifacts = run_simulation_analysis(config, INTERACTION, stages=("efficiencies",))

    assert artifacts.stage_status["efficiencies"] == "completed"
    mock_assign.assert_called_once()
    assert mock_compute.call_args.kwargs["vertex_counts"] == {200: COUNTED}


def test_new_energy_is_counted_while_stored_counts_are_reused(
    tmp_path: Path, mock_assign: MagicMock, mock_compute: MagicMock
) -> None:
    original_config = _make_config(tmp_path)
    config = replace(
        original_config,
        energies_keV=(200, 300),
        fep_window=FepWindowConfig(half_width_fwhm=1.5),
    )
    _materialize_cached_inputs(config)
    dataset_path, _, _ = _simulation_paths(config)
    _touch(dataset_path / "sim_e=300/data.parquet")
    _write_simulation_manifest(
        config, efficiency_setup=current_cut_setup(original_config)
    )
    _materialize_simulation_inputs(config, energy=300)
    _materialize_calibration_inputs(config)

    artifacts = run_simulation_analysis(config, INTERACTION, stages=("efficiencies",))

    assert artifacts.stage_status["efficiencies"] == "completed"
    # Only the 300 keV STP files are opened.
    mock_assign.assert_called_once()
    assert mock_assign.call_args.kwargs["lh5_files"] == [
        _job_file(config, "stp", 300, "000")
    ]
    assert mock_compute.call_args.kwargs["simulated_energies"] == [200, 300]
    assert mock_compute.call_args.kwargs["vertex_counts"] == {
        200: CACHED_COUNTS,
        300: COUNTED,
    }
    assert _read_manifest(config)["stages"]["efficiencies"]["primary_counts"] == {
        "reused_energies_keV": [200],
        "counted_energies_keV": [300],
    }


def test_mismatched_stp_and_evt_jobs_block_the_efficiency(
    tmp_path: Path, mock_assign: MagicMock, mock_compute: MagicMock
) -> None:
    config = _make_config(tmp_path)
    _, partition_path, _ = _simulation_paths(config)
    _touch(partition_path)
    _materialize_simulation_inputs(config, stp_jobs=("000",), evt_jobs=("000", "001"))
    _materialize_calibration_inputs(config)

    artifacts = run_simulation_analysis(config, INTERACTION, stages=("efficiencies",))

    assert artifacts.stage_status["efficiencies"] == "blocked"
    assert any(
        "STP and evt jobs differ (001)" in warning for warning in artifacts.warnings
    )
    mock_assign.assert_not_called()
    mock_compute.assert_not_called()


@patch(f"{SIMULATION}.build_parquet_dataset")
def test_recomputed_dataset_marks_downstream_stages_stale(
    mock_build: MagicMock,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    _materialize_cached_inputs(config)
    _, partition_path, _ = _simulation_paths(config)
    partition_path.unlink()
    _write_simulation_manifest(
        config,
        efficiency_setup=current_cut_setup(config),
        include_plots=True,
    )
    _touch(_job_file(config, "evt", 200, "000"))

    def create_partition(**_kwargs: object) -> None:
        partition_path.parent.mkdir(parents=True, exist_ok=True)
        partition_path.write_bytes(b"rebuilt")

    mock_build.side_effect = create_partition
    run_simulation_analysis(config, INTERACTION, stages=("build-dataset",))

    manifest = _read_manifest(config)
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

    with patch(f"{SIMULATION}.build_analysis_context") as mock_context:
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
        patch(f"{SIMULATION}.build_analysis_context") as mock_context,
        patch(
            f"{SIMULATION}.build_labels_dicts",
            return_value={},
        ),
        patch(
            f"{SIMULATION}.plot_efficiency_comparison",
            return_value=(object(), None),
        ) as mock_efficiency_plot,
        patch(
            f"{SIMULATION}.plot_fep_survival_fraction",
            return_value=(object(), None),
        ) as mock_survival_plot,
        patch(f"{SIMULATION}.plt.close"),
    ):
        mock_context.return_value.eres_dict = {}
        artifacts = run_simulation_analysis(config, INTERACTION, stages=("plots",))

    assert artifacts.stage_status == {
        "build-dataset": "cached",
        "efficiencies": "cached",
        "plots": "completed",
    }
    assert mock_efficiency_plot.call_count == 4
    assert mock_survival_plot.call_count == 2

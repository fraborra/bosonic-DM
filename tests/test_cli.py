# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bosonic_dm.cli import _banner_rows, _render_banner, main
from bosonic_dm.models import AnalysisArtifacts
from bosonic_dm.pipeline.background import BACKGROUND_DEFAULT_STAGES


@pytest.fixture
def mock_artifacts():
    return AnalysisArtifacts(
        dataset_paths=[],
        yaml_paths=[],
        plot_paths=[],
    )


@patch("bosonic_dm.cli.load_analysis_config")
@patch("bosonic_dm.cli.run_simulation_analysis")
def test_simulation_command_valid(mock_run, mock_load, mock_artifacts):
    """Test valid simulation command passes arguments to the pipeline."""
    mock_config = MagicMock()
    mock_load.return_value = mock_config
    mock_run.return_value = mock_artifacts

    ret = main(
        ["simulation", "--config", "dummy.yaml", "--interaction", "axio-electric"]
    )
    assert ret == 0

    mock_load.assert_called_once()
    mock_run.assert_called_once()

    kwargs = mock_run.call_args[1]
    assert kwargs["interaction"] == "axio-electric"
    assert kwargs["overwrite"] is None


def test_missing_config():
    """Test missing config raises SystemExit via argparse."""
    with pytest.raises(SystemExit):
        main(["simulation", "--interaction", "axio-electric"])


@patch("bosonic_dm.cli.load_analysis_config")
def test_invalid_config(mock_load):
    """Test error during config loading exits via argparse error."""
    mock_load.side_effect = ValueError("Invalid config")
    with pytest.raises(SystemExit):
        main(["simulation", "--config", "dummy.yaml", "--interaction", "axio-electric"])


@patch("bosonic_dm.cli.load_analysis_config")
@patch("bosonic_dm.cli.run_simulation_analysis")
def test_stage_handling(mock_run, mock_load, mock_artifacts):
    """Test --stage passes specific stages to the pipeline."""
    mock_load.return_value = MagicMock()
    mock_run.return_value = mock_artifacts

    ret = main(
        [
            "simulation",
            "--config",
            "dummy.yaml",
            "--interaction",
            "axio-electric",
            "--stage",
            "build-dataset",
            "plots",
        ]
    )
    assert ret == 0

    kwargs = mock_run.call_args[1]
    assert kwargs["stages"] == ["build-dataset", "plots"]


@patch("bosonic_dm.cli.load_analysis_config")
@patch("bosonic_dm.cli.run_simulation_analysis")
def test_overwrite_handling(mock_run, mock_load, mock_artifacts):
    """Test --overwrite passes True to the pipeline."""
    mock_load.return_value = MagicMock()
    mock_run.return_value = mock_artifacts

    ret = main(
        [
            "simulation",
            "--config",
            "dummy.yaml",
            "--interaction",
            "axio-electric",
            "--overwrite",
        ]
    )
    assert ret == 0

    kwargs = mock_run.call_args[1]
    assert kwargs["overwrite"] is True


@patch("bosonic_dm.cli.load_analysis_config")
@patch("bosonic_dm.cli.run_background_analysis")
def test_background_command(mock_run, mock_load, mock_artifacts):
    """Test valid background command passes arguments to the pipeline."""
    mock_load.return_value = MagicMock()
    mock_run.return_value = mock_artifacts

    ret = main(["background", "--config", "dummy.yaml"])
    assert ret == 0

    mock_run.assert_called_once()

    kwargs = mock_run.call_args[1]
    assert kwargs["overwrite"] is None
    assert kwargs["stages"] == BACKGROUND_DEFAULT_STAGES


def _banner_config() -> MagicMock:
    """Return a concrete-enough config mock for banner rendering tests."""
    config = MagicMock()
    config.output.overwrite = True
    config.output.save_plots = True
    config.output.write_manifest = True
    config.paths.simulation_root = "/simulation"
    config.paths.data_root = "data/v4"
    config.energies_keV = tuple(range(100, 1030, 10))
    config.selections = ("all", "valid-psd", "sse", "mse")
    config.fep_window.half_width_fwhm = 2.0
    config.detector_groups = "data/inputs/groups.yaml"
    config.background.pet_glob = "/data/pet/phy/*.lh5"
    config.background.comparison_cut_profile = "without-bb-like"
    config.background.energy_ranges_keV = [(20, 300), (20, 1000)]
    config.background.bin_widths_keV = [1, 5, 10]
    config.apply_lar_veto = True
    return config


def test_background_banner_excludes_simulation_settings() -> None:
    rows = _banner_rows(
        "background",
        _banner_config(),
        interactions=None,
        stages=("build-dataset",),
        overwrite=None,
    )
    labels = {label for label, _ in rows}

    assert "PET inputs" in labels
    assert "Comparison cut" in labels
    assert "Energies [keV]" not in labels
    assert "Selections" not in labels
    assert "FEP half-width" not in labels
    assert dict(rows)["Overwrite"] == "True"


def test_simulation_banner_excludes_background_settings() -> None:
    rows = _banner_rows(
        "simulation",
        _banner_config(),
        interactions=("axio-electric",),
        stages=("efficiencies",),
        overwrite=False,
    )
    labels = {label for label, _ in rows}

    assert "Energies [keV]" in labels
    assert "Selections" in labels
    assert "PET inputs" not in labels
    assert "Comparison cut" not in labels
    assert "Energy ranges [keV]" not in labels
    assert dict(rows)["Overwrite"] == "False"


def test_banner_wraps_long_values_to_fixed_width() -> None:
    rows = _banner_rows(
        "simulation",
        _banner_config(),
        interactions=("axio-electric",),
        stages=("plots",),
        overwrite=None,
    )
    banner = _render_banner(rows)

    assert max(len(line) for line in banner.splitlines()) == 80
    assert len(banner.splitlines()) > len(rows) + 4

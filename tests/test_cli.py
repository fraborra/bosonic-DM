# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bosonic_dm.cli import main
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

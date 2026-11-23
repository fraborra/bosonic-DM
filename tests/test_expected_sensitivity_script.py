# Copyright (C) 2026 Francesco Borra
#

"""``scripts/expected_sensitivity.py`` and ``load_signal_inputs`` on minimal outputs."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from signal_pipeline_outputs import (
    B_HAT,
    FIT_EDGES_KEV,
    OBSERVED,
    write_signal_pipeline_outputs,
)

from bosonic_dm.config import AnalysisConfig
from bosonic_dm.fitting.sensitivity import draw_background_toys
from bosonic_dm.fitting.signal_template import SignalWindow
from bosonic_dm.pipeline.background import background_fit_dir
from bosonic_dm.pipeline.sensitivity import load_signal_inputs
from bosonic_dm.yaml_io import read_yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "expected_sensitivity.py"
_spec = importlib.util.spec_from_file_location("expected_sensitivity", SCRIPT)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

SEED = 7


@pytest.fixture
def config(tmp_path: Path) -> AnalysisConfig:
    return write_signal_pipeline_outputs(tmp_path)


def test_load_signal_inputs_reads_the_pipeline_outputs(config: AnalysisConfig) -> None:
    inputs = load_signal_inputs(config)
    assert list(inputs.backgrounds) == ["B"]
    assert inputs.fit_dirs == {"B": background_fit_dir(config, "B")}
    assert inputs.efficiency_table.is_file()
    np.testing.assert_allclose(inputs.backgrounds["B"].b_hat, B_HAT)
    assert set(inputs.window_options) == {
        "eres_dict",
        "groups_dict",
        "group_efficiencies",
    }


def test_load_signal_inputs_rejects_unknown_groups(config: AnalysisConfig) -> None:
    with pytest.raises(ValueError, match="Groups not in"):
        load_signal_inputs(config, groups=["C"])


def _run(tmp_path: Path, *extra: str) -> Path:
    outdir = tmp_path / "sensitivity"
    script.main(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--masses",
            "290,300,310",
            "--n-toys",
            "5",
            "--seed",
            str(SEED),
            "--outdir",
            str(outdir),
            *extra,
        ]
    )
    return outdir


def test_main_writes_outputs_and_records_the_config(
    config: AnalysisConfig, tmp_path: Path
) -> None:
    outdir = _run(tmp_path, "--window-plot-masses", "300")
    for name in (
        "limits_per_toy.parquet",
        "expected_limits.parquet",
        "expected_limits.yaml",
        "expected_limits.png",
        "window_300keV.png",
    ):
        assert (outdir / name).is_file(), name
    record = read_yaml(outdir / "expected_limits.yaml")
    assert record["inputs"]["config"] == str(tmp_path / "config.yaml")
    assert record["inputs"]["interaction"] == "axio-electric"
    assert record["inputs"]["background_fit_dirs"] == {
        "B": str(background_fit_dir(config, "B"))
    }
    assert record["masses_keV"]["n"] == 3


@pytest.mark.usefixtures("config")  # writes the inputs and config.yaml
def test_window_plot_shows_toy_0_not_the_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plotted: list[SignalWindow] = []

    def fake_plot(window: SignalWindow, _rate: float, **kwargs: Any) -> Any:
        plotted.append(window)
        assert kwargs["counts_label"] == "toy 0"
        assert kwargs.get("counts") is None
        return script.plt.figure()

    monkeypatch.setattr(script, "plot_window_fit", fake_plot)
    _run(tmp_path, "--window-plot-masses", "300")

    assert len(plotted) == 1
    window = plotted[0]
    full_toy = draw_background_toys(
        np.full((1, len(FIT_EDGES_KEV) - 1), B_HAT), 1, SEED
    )[0]
    lo = int(np.searchsorted(FIT_EDGES_KEV, window.bin_edges[0]))
    expected = full_toy[:, lo : lo + len(window.bin_edges) - 1]
    np.testing.assert_array_equal(window.stacked("counts"), expected)
    assert not np.all(window.stacked("counts") == OBSERVED)


def test_parse_masses_mixes_numbers_and_ranges() -> None:
    masses = script.parse_masses("1020,123,150,200:220:10,150")
    np.testing.assert_allclose(masses, [123, 150, 200, 210, 220, 1020])
    np.testing.assert_allclose(script.parse_masses("100:102:1"), [100, 101, 102])
    assert len(script.parse_masses("123,150,175,200:400:10,450:1000:50,1020")) == 37

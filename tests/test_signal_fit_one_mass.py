# Copyright (C) 2026 Francesco Borra
#

"""``scripts/signal_fit_one_mass.py`` on a minimal set of pipeline outputs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from signal_pipeline_outputs import B_HAT, OBSERVED, write_signal_pipeline_outputs

from bosonic_dm.config import AnalysisConfig
from bosonic_dm.fitting.signal_template import SignalWindow
from bosonic_dm.plotting.signal_fit import plot_window_fit
from bosonic_dm.yaml_io import read_yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "signal_fit_one_mass.py"
_spec = importlib.util.spec_from_file_location("signal_fit_one_mass", SCRIPT)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

MASS = 300.0


@pytest.fixture
def config(tmp_path: Path) -> AnalysisConfig:
    """Config, groups, eres, efficiency table and one background fit on disk."""
    return write_signal_pipeline_outputs(tmp_path)


@pytest.fixture
def window(config: AnalysisConfig) -> SignalWindow:
    return script.load_window(config, MASS)


def test_load_window_reads_the_pipeline_outputs(window: SignalWindow) -> None:
    assert window.group_names == ["B"]
    assert len(window.bin_edges) - 1 == 24
    inputs = window.groups["B"]
    assert np.all(inputs.counts == OBSERVED)
    np.testing.assert_allclose(inputs.b_hat, B_HAT)
    assert inputs.exposure_kg_yr == pytest.approx(6.0)
    assert inputs.efficiency == pytest.approx(0.5)


def test_load_window_rejects_unknown_groups(config: AnalysisConfig) -> None:
    with pytest.raises(ValueError, match="Groups not in"):
        script.load_window(config, MASS, groups=["C"])


def test_blinded_check_never_fits_the_observed_counts(
    window: SignalWindow, signal_model: object, tmp_path: Path
) -> None:
    seed = 3
    checks = script.check_mass(window, signal_model, tmp_path, seed=seed)

    toy = np.random.default_rng(seed).poisson(window.stacked("b_hat"))
    record = read_yaml(tmp_path / "summary.yaml")
    fitted = record["groups"]["B"]["observed_counts_fit_bins"]
    assert fitted == int(toy.sum())
    assert fitted != OBSERVED * 24
    assert record["provenance"]["dataset"] == {
        "kind": "toy",
        "seed": seed,
        "injected_rate": 0.0,
    }
    assert checks["stan"]["passed_diagnostics"]
    assert checks["stan_grid_agree"], checks
    assert checks["asimov_upper_limit_grid"] > 0
    assert read_yaml(tmp_path / "checks.yaml")["dataset"]["kind"] == "toy"
    for name in ("rate_draws.npy", "window.png"):
        assert (tmp_path / name).exists()


def test_injected_signal_is_recovered(
    window: SignalWindow, signal_model: object, tmp_path: Path
) -> None:
    checks = script.check_mass(
        window, signal_model, tmp_path, seed=11, inject_rate=40.0
    )
    assert checks["stan"]["passed_diagnostics"]
    assert abs(checks["injection_pull"]) < 3


def test_unblind_fits_the_observed_counts(
    window: SignalWindow, signal_model: object, tmp_path: Path
) -> None:
    checks = script.check_mass(window, signal_model, tmp_path, seed=5, unblind=True)
    record = read_yaml(tmp_path / "summary.yaml")
    assert record["groups"]["B"]["observed_counts_fit_bins"] == OBSERVED * 24
    assert checks["dataset"] == {"kind": "observed"}
    with pytest.raises(ValueError, match="needs a toy"):
        script.check_mass(
            window, signal_model, tmp_path, seed=5, unblind=True, inject_rate=1.0
        )


@pytest.mark.usefixtures("config")  # writes the inputs and config.yaml
def test_main_runs_blinded_by_default(
    signal_model: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Reuse the model compiled in a temporary directory.
    monkeypatch.setattr(script, "SIGNAL_STAN_FILE", Path(signal_model.stan_file))
    outdir = tmp_path / "out"
    with caplog.at_level("INFO", logger="signal_fit_one_mass"):
        script.main(
            [
                "--config",
                str(tmp_path / "config.yaml"),
                "--mass",
                str(MASS),
                "--outdir",
                str(outdir),
                "--inject",
                "40",
            ]
        )
    text = caplog.text
    assert "blinded: toy counts" in text
    assert "Asimov R90 (grid)" in text
    assert "pull" in text
    record = read_yaml(outdir / "summary.yaml")
    assert record["provenance"]["dataset"]["kind"] == "toy"
    assert record["provenance"]["interaction"] == "axio-electric"
    assert (outdir / "window.png").exists()


def test_inject_and_unblind_are_exclusive(tmp_path: Path) -> None:
    base = ["--config", "c.yaml", "--mass", "500", "--outdir", str(tmp_path)]
    assert not script.parse_args(base).unblind
    with pytest.raises(SystemExit):
        script.parse_args([*base, "--unblind", "--inject", "1"])
    with pytest.raises(SystemExit):
        script.parse_args([*base, "--inject", "-1"])


def test_plot_window_fit(window: SignalWindow, tmp_path: Path) -> None:
    path = tmp_path / "window.png"
    toy = window.with_counts(np.zeros((1, 24), dtype=int))
    fig = plot_window_fit(toy, 2.0, counts_label="toy", output_path=path)
    assert path.exists()
    assert len(fig.axes) == len(window.groups)
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert "toy" in labels
    with pytest.raises(ValueError, match="counts must have shape"):
        plot_window_fit(window, 2.0, counts=np.zeros((1, 3)))
    plt.close("all")

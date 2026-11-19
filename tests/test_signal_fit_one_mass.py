# Copyright (C) 2026 Francesco Borra
#

"""``scripts/signal_fit_one_mass.py`` on a minimal set of pipeline outputs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
import yaml

from bosonic_dm.config import AnalysisConfig, load_analysis_config
from bosonic_dm.efficiency import (
    efficiency_table,
    efficiency_table_path,
    write_efficiency_table,
)
from bosonic_dm.fitting.signal_template import SignalWindow
from bosonic_dm.pipeline.background import background_fit_dir
from bosonic_dm.plotting.signal_fit import plot_window_fit
from bosonic_dm.yaml_io import read_yaml, write_yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "signal_fit_one_mass.py"
_spec = importlib.util.spec_from_file_location("signal_fit_one_mass", SCRIPT)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

MASS = 300.0
OBSERVED = 9  # observed counts per bin; a blinded check must never fit them
B_HAT = 6.5
GROUPS = {"B": {"D4": "all"}}


def _det(expo: float) -> dict:
    return {
        "usability": "on",
        "expo": expo,
        "a": 1.5,
        "b": 0.002,
        "a_unc": 0.0,
        "b_unc": 0.0,
        "ab_corr": 0.0,
    }


def _efficiency_entry(efficiency: float, expo: float) -> dict:
    n_primaries = round(1000 * expo)
    return {
        "expo": expo,
        "n_primaries": n_primaries,
        "selections": {
            "all": {
                "status": "valid",
                "n_events": round(efficiency * n_primaries),
                "efficiency": efficiency,
                "efficiency_stat_unc": 0.01,
                "efficiency_syst_fwhm": 0.0,
            }
        },
    }


@pytest.fixture
def config(tmp_path: Path) -> AnalysisConfig:
    """Config, groups, eres, efficiency table and one background fit on disk."""
    inputs = tmp_path / "inputs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
schema_version: 1
production:
  version: v1
  reference_root: {tmp_path}/production
paths:
  simulation_root: {tmp_path}/simulation
  data_root: {tmp_path}/outputs
  inputs_root: {inputs}
analysis:
  simulated_energies_keV: [200, 400]
  fep_window:
    half_width_fwhm: 2.0
  selections: [all]
  apply_lar_veto: true
interactions:
  axio-electric:
    job_template: electron_{{energy}}keV_hpge_bulk
background:
  comparison_cut_profile: without-bb-like
  energy_ranges_keV: [[20, 300]]
  bin_widths_keV: [5]
output: {{}}
""",
        encoding="utf-8",
    )
    config = load_analysis_config(config_path)

    config.detector_groups.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(GROUPS, config.detector_groups)
    # safe_dump quotes "on", as in the real eres_dict.yaml; unquoted it would
    # be read back as a boolean.
    eres = {period: {"r000": {"D4": _det(3.0)}} for period in ("p03", "p04")}
    (inputs / "dictionaries" / "eres_dict.yaml").write_text(
        yaml.safe_dump(eres), encoding="utf-8"
    )

    efficiency = {
        energy: {
            "D4": {
                "period_runs": {
                    period: {"r000": _efficiency_entry(0.5, 3.0)}
                    for period in ("p03", "p04")
                }
            }
        }
        for energy in (200, 400)
    }
    path = efficiency_table_path(config.paths.dictionaries_root, "axio-electric")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_efficiency_table(efficiency_table(efficiency), path)

    fit_dir = background_fit_dir(config, "B")
    fit_dir.mkdir(parents=True)
    edges = np.arange(250.0, 351.0)
    pl.DataFrame(
        {
            "bin_lo_keV": edges[:-1],
            "bin_hi_keV": edges[1:],
            "counts": np.full(len(edges) - 1, OBSERVED),
            "B_hat": np.full(len(edges) - 1, B_HAT),
            "used_in_fit": np.ones(len(edges) - 1, dtype=bool),
        }
    ).write_parquet(fit_dir / "bhat.parquet")
    write_yaml(
        {"passed_diagnostics": True, "provenance": {"group": "B"}},
        fit_dir / "summary.yaml",
    )
    return config


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

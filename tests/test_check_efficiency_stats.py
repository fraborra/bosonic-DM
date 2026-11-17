# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import pytest

from bosonic_dm.check_efficiency_stats import efficiency_stat_table, main
from bosonic_dm.efficiency import (
    efficiency_table,
    efficiency_table_path,
    write_efficiency_table,
)
from bosonic_dm.stats import bayesian_efficiency

INTERACTION = "axio-electric"


def _run(expo: float, n_events: int, n_primaries: int) -> dict:
    efficiency, stat_unc = bayesian_efficiency(n_events, n_primaries)
    return {
        "expo": expo,
        "n_primaries": n_primaries,
        "selections": {
            "all": {
                "status": "valid",
                "n_events": n_events,
                "efficiency": efficiency,
                "efficiency_stat_unc": stat_unc,
            },
            "sse": {"status": "psd-unavailable"},
        },
    }


def _table() -> pl.DataFrame:
    return efficiency_table(
        {
            200: {
                # Two runs of one detector: weights 1/4 and 3/4.
                "V00001A": {
                    "period_runs": {
                        "p01": {"r001": _run(1.0, 800, 1000)},
                        "p02": {"r001": _run(3.0, 300, 500)},
                    },
                },
                "B00002A": {
                    "period_runs": {"p01": {"r001": _run(2.0, 50, 100)}},
                },
            },
            300: {
                "V00001A": {
                    "period_runs": {"p01": {"r001": _run(1.0, 9000, 10000)}},
                },
            },
        }
    )


def test_stat_table_combines_run_counts_with_exposure_weights() -> None:
    stats = efficiency_stat_table(_table())

    assert stats.select("sim_e", "detector").rows() == [
        (200, "B00002A"),
        (200, "V00001A"),
        (300, "V00001A"),
    ]
    row = stats.row(1, named=True)
    assert row["detector_type"] == "ICPC"
    # 800/1000 with weight 1/4 and 300/500 with weight 3/4: mean 0.65, and
    # one Jeffreys prior on the effective primaries of the detector.
    effective_primaries = 1 / (0.25**2 / 1000 + 0.75**2 / 500)
    efficiency, stat_unc = bayesian_efficiency(
        0.65 * effective_primaries, effective_primaries
    )
    assert row["efficiency"] == pytest.approx(efficiency)
    assert row["efficiency"] == pytest.approx(0.65, abs=1e-3)
    assert row["efficiency_stat_unc"] == pytest.approx(stat_unc)
    assert row["relative_unc"] == pytest.approx(stat_unc / efficiency)
    assert row["n_primaries"] == 1500
    assert row["n_events"] == 1100
    assert stats.row(0, named=True)["detector_type"] == "BEGe"


def test_stat_table_skips_selections_without_a_valid_result() -> None:
    assert efficiency_stat_table(_table(), selection="sse").is_empty()


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
schema_version: 1
production:
  version: v1
  reference_root: {tmp_path}/production
  metadata_override: null
paths:
  simulation_root: {tmp_path}/simulation
  data_root: {tmp_path}/data
  inputs_root: {tmp_path}/inputs
  plots_root: {tmp_path}/plots
  temporary_root: {tmp_path}/tmp
analysis:
  simulated_energies_keV: [200, 300]
  fep_window:
    half_width_fwhm: 2.0
  selections: [all]
  apply_lar_veto: true
  detector_groups: {tmp_path}/groups.yaml
interactions:
  {INTERACTION}:
    job_template: electron_{{energy}}keV_hpge_bulk
background:
  pet_glob: {tmp_path}/pet/*.lh5
  comparison_cut_profile: without-bb-like
  energy_ranges_keV: [[20, 300]]
  bin_widths_keV: [5]
output:
  overwrite: false
  save_plots: true
  write_manifest: true
""",
        encoding="utf-8",
    )
    return config_path


def test_main_reports_the_worst_detectors(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config_path = _write_config(tmp_path)
    write_efficiency_table(
        _table(), efficiency_table_path(tmp_path / "data/dictionaries", INTERACTION)
    )

    with caplog.at_level(logging.INFO, logger="bosonic_dm.check_efficiency_stats"):
        status = main(["--config", str(config_path), "--interaction", INTERACTION])

    assert status == 0
    assert "200 keV: 2 of 2 detectors above the 1.00% target" in caplog.text
    assert "300 keV: 0 of 1 detectors above the 1.00% target" in caplog.text
    # B00002A (50 in 100) has 9.9% relative uncertainty: 98 times the
    # primaries for 1%.
    assert "about 98.0x its simulated primaries" in caplog.text
    messages = caplog.messages
    heading = messages.index("  Largest relative statistical uncertainty:")
    assert messages[heading + 1].strip().startswith("B00002A (BEGe)")


def test_main_requires_an_existing_efficiency_table(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    with pytest.raises(SystemExit):
        main(["--config", str(config_path), "--interaction", INTERACTION])
    with pytest.raises(SystemExit):
        main(["--config", str(config_path), "--interaction", "unknown"])

# Copyright (C) 2026 Francesco Borra
#

"""Minimal pipeline outputs read by the signal-fit and sensitivity scripts.

One group ``B`` (detector ``D4``), a background fit on 250-350 keV in 1 keV
bins with flat ``B_hat`` and flat observed counts, an efficiency table at 200
and 400 keV and a resolution dictionary, all under ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import yaml

from bosonic_dm.config import AnalysisConfig, load_analysis_config
from bosonic_dm.efficiency import (
    efficiency_table,
    efficiency_table_path,
    write_efficiency_table,
)
from bosonic_dm.pipeline.background import background_fit_dir
from bosonic_dm.yaml_io import write_yaml

OBSERVED = 9  # observed counts per bin; a blinded script must never use them
B_HAT = 6.5
GROUPS = {"B": {"D4": "all"}}
FIT_EDGES_KEV = np.arange(250.0, 351.0)


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


def write_signal_pipeline_outputs(tmp_path: Path) -> AnalysisConfig:
    """Write ``config.yaml`` and the outputs it names; return the config."""
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
    edges = FIT_EDGES_KEV
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

# Copyright (C) 2025 Francesco Borra
#

from __future__ import annotations

from pathlib import Path

from bosonic_dm.config import load_analysis_config


def test_calibration_inputs_are_separate_from_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    calibration_root = tmp_path / "calibration"
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
  data_root: {output_root}
  calibration_dictionaries_root: {calibration_root}
  plots_root: {tmp_path}/plots
  temporary_root: {tmp_path}/tmp
analysis:
  simulated_energies_keV: [200]
  fep_window:
    half_width_fwhm: 2.0
  selections: [all]
interactions:
  axio-electric:
    job_template: electron_{{energy}}keV_hpge_bulk
    make_lar_survival_plots: false
background:
  pet_glob: {tmp_path}/pet/*.lh5
  apply_lar_veto: true
  comparison_cut_profile: without-bb-like
  energy_ranges_keV: [[20, 300]]
  bin_widths_keV: [5]
  detector_groups: {tmp_path}/groups.yaml
output:
  overwrite: false
  save_plots: true
  show_plots: false
  write_manifest: true
""",
        encoding="utf-8",
    )

    config = load_analysis_config(config_path)

    assert config.paths.dictionaries_root == output_root / "dictionaries"
    assert config.paths.calibration_dictionaries_root == calibration_root

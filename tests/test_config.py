# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

from pathlib import Path

import pytest

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
  inputs_root: {calibration_root}
  plots_root: {tmp_path}/plots
  temporary_root: {tmp_path}/tmp
analysis:
  simulated_energies_keV: [200]
  fep_window:
    half_width_fwhm: 2.0
  selections: [all]
  apply_lar_veto: true
  detector_groups: {tmp_path}/groups.yaml
interactions:
  axio-electric:
    job_template: electron_{{energy}}keV_hpge_bulk
    make_lar_survival_plots: false
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

    config = load_analysis_config(config_path)

    assert config.paths.dictionaries_root == output_root / "dictionaries"
    assert config.paths.inputs_root == calibration_root
    assert config.apply_lar_veto is True


def _write_config(tmp_path: Path, substitutes: str) -> Path:
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
analysis:
  simulated_energies_keV: [200]
  fep_window:
    half_width_fwhm: 2.0
  selections: [all]
  apply_lar_veto: true
{substitutes}
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
    return config_path


def test_run_substitutes_are_read_and_recorded_in_the_cut_setup(
    tmp_path: Path,
) -> None:
    from bosonic_dm.pipeline.manifest import current_cut_setup  # noqa: PLC0415

    config = load_analysis_config(
        _write_config(tmp_path, "  efficiency_run_substitutes:\n    p09-r004: p09-r005")
    )

    assert config.efficiency_run_substitutes == {("p09", "r004"): ("p09", "r005")}
    assert current_cut_setup(config)["efficiency_run_substitutes"] == {
        "p09-r004": "p09-r005"
    }


def test_run_substitutes_default_to_none(tmp_path: Path) -> None:
    config = load_analysis_config(_write_config(tmp_path, ""))

    assert config.efficiency_run_substitutes == {}


@pytest.mark.parametrize(
    ("mapping", "match"),
    [
        ("p09-r004: p09-r004", "itself"),
        ("p09-r004: p09-r005\n    p09-r005: p09-r006", "itself replaced"),
        ("p9-r4x: p09-r005", "period-run label"),
    ],
)
def test_invalid_run_substitutes_are_rejected(
    tmp_path: Path, mapping: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        load_analysis_config(
            _write_config(tmp_path, f"  efficiency_run_substitutes:\n    {mapping}")
        )

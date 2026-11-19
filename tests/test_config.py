# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

from pathlib import Path

import pytest

from bosonic_dm.config import (
    DEFAULT_BACKGROUND_FIT_PEAKS_KEV,
    BackgroundFitConfig,
    load_analysis_config,
)


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


def _write_config(tmp_path: Path, substitutes: str, background_fit: str = "") -> Path:
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
{background_fit}
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


def test_background_fit_defaults_when_section_is_absent(tmp_path: Path) -> None:
    config = load_analysis_config(_write_config(tmp_path, ""))

    fit = config.background.fit
    assert fit.spectrum_range_keV == (60.0, 2000.0)
    assert fit.spectrum_bin_width_keV == 1.0
    assert fit.fit_range_keV == (100.0, 1100.0)
    # Standard table, restricted to the 10 lines inside 100-1100 keV.
    assert fit.peaks_keV == tuple(
        c for c in DEFAULT_BACKGROUND_FIT_PEAKS_KEV if 95.0 < c < 1105.0
    )
    assert len(fit.peaks_keV) == 10
    assert fit.peak_exclusion_width_keV == 5.0
    assert fit.prefit_dictionary == (
        config.paths.inputs_root / "dictionaries/prefit_dictionary.yaml"
    )
    assert fit.prefit_key == "fit_ar"
    assert fit.model == "no_linear"
    assert fit.shape_sd == 0.5
    assert fit.rate_prior_sd == {"n2nu": 0.5, "ar39": 2.0}
    assert fit.sampler.chains == 4
    assert fit.sampler.iter_warmup == 1000
    assert fit.sampler.iter_sampling == 1000
    assert fit.sampler.seed == 12345


def test_background_fit_section_overrides_defaults(tmp_path: Path) -> None:
    section = """  fit:
    fit_range_keV: [120, 1000]
    prefit_dictionary: /abs/prefit.yaml
    model: full
    rate_prior_sd: {n2nu: 1.0}
    sampler: {chains: 2}"""
    config = load_analysis_config(_write_config(tmp_path, "", section))

    fit = config.background.fit
    assert fit.fit_range_keV == (120.0, 1000.0)
    assert fit.prefit_dictionary == Path("/abs/prefit.yaml")
    assert fit.model == "full"
    assert fit.rate_prior_sd == {"n2nu": 1.0, "ar39": 2.0}
    assert fit.sampler.chains == 2
    assert fit.sampler.seed == 12345


@pytest.mark.parametrize(
    ("peaks", "expected"),
    [
        ("standard", (238.632, 295.224, 351.932, 511.0, 583.187)),
        ("[1460.82, 583.187, 511.0, 583.187]", (511.0, 583.187)),
        ("[97, 603]", (97.0, 603.0)),
        ("[90, 610]", ()),
        ("[]", ()),
    ],
)
def test_background_fit_peaks_standard_or_list(
    tmp_path: Path, peaks: str, expected: tuple[float, ...]
) -> None:
    section = f"  fit:\n    fit_range_keV: [100, 600]\n    peaks_keV: {peaks}"
    config = load_analysis_config(_write_config(tmp_path, "", section))

    assert config.background.fit.peaks_keV == expected


@pytest.mark.parametrize(
    ("section", "match"),
    [
        ("  fit:\n    model: linear", "background.fit.model"),
        ("  fit:\n    peaks_keV: all", "background.fit.peaks_keV"),
        ("  fit:\n    peaks_keV: [583.187, Tl-208]", "not a number"),
        ("  fit:\n    shapesd: 0.5", "Unknown background.fit keys"),
        ("  fit:\n    sampler: {thin: 2}", "Unknown background.fit.sampler keys"),
    ],
)
def test_invalid_background_fit_section_is_rejected(
    tmp_path: Path, section: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        load_analysis_config(_write_config(tmp_path, "", section))


def test_background_fit_config_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="no_linear"):
        BackgroundFitConfig(model="linear")

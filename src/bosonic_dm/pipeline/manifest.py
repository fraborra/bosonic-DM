# Copyright (C) 2026 Francesco Borra
#

"""Current-state manifest helpers shared by analysis pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from bosonic_dm.config import AnalysisConfig
from bosonic_dm.efficiency import build_selection_metadata
from bosonic_dm.yaml_io import read_yaml

MANIFEST_SCHEMA_VERSION = 2
EFFICIENCY_OUTPUT_SCHEMA_VERSION = 2
BACKGROUND_DATASET_SCHEMA_VERSION = 1
BACKGROUND_STORED_FLAGS = (
    "passes_baseline",
    "passes_default",
    "passes_without_bb_like",
    "passes_lar",
    "passes_analysis",
    "passes_comparison",
)


def _load_manifest(path: Path) -> dict[str, object]:
    """Load a manifest mapping, returning an empty mapping when unavailable."""
    if not path.exists():
        return {}
    loaded = read_yaml(path)
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def load_simulation_manifest(path: Path, interaction: str) -> dict[str, object]:
    """Load one simulation manifest and normalize its current-state envelope."""
    loaded = _load_manifest(path)
    if (
        loaded.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or loaded.get("interaction") != interaction
    ):
        loaded = {}
    stages = loaded.get("stages")
    return {
        **loaded,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "interaction": interaction,
        "stages": dict(stages) if isinstance(stages, Mapping) else {},
    }


def load_background_manifest(path: Path) -> dict[str, object]:
    """Load the background manifest and normalize its current-state envelope."""
    loaded = _load_manifest(path)
    if loaded.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        loaded = {}
    stages = loaded.get("stages")
    return {
        **loaded,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stages": dict(stages) if isinstance(stages, Mapping) else {},
    }


def current_cut_setup(config: AnalysisConfig) -> dict[str, object]:
    """Return the settings that determine simulation efficiency products."""
    return {
        "fep_half_width_fwhm": config.fep_window.half_width_fwhm,
        "selections": list(config.selections),
        "apply_lar_veto": config.apply_lar_veto,
        "selection_metadata": build_selection_metadata(
            config.selections,
            half_width_fwhm=config.fep_window.half_width_fwhm,
            apply_lar_veto=config.apply_lar_veto,
        ),
        "efficiency_output_schema_version": EFFICIENCY_OUTPUT_SCHEMA_VERSION,
    }


def current_plot_setup(
    config: AnalysisConfig,
    interaction: str,
) -> dict[str, object]:
    """Return the settings that determine simulation plot products."""
    interaction_config = config.interactions[interaction]
    return {
        "detector_groups": str(config.detector_groups),
        "make_energy_spectra_plots": (interaction_config.make_energy_spectra_plots),
        "make_lar_survival_plots": interaction_config.make_lar_survival_plots,
        "make_aoe_survival_plots": interaction_config.make_aoe_survival_plots,
    }


def current_background_cut_setup(config: AnalysisConfig) -> dict[str, object]:
    """Return the settings that determine stored background cut flags."""
    return {
        "multiplicity": "one",
        "apply_lar_veto": config.apply_lar_veto,
        "comparison_cut_profile": config.background.comparison_cut_profile,
        "dataset_schema_version": BACKGROUND_DATASET_SCHEMA_VERSION,
        "stored_flags": list(BACKGROUND_STORED_FLAGS),
    }


def current_background_plot_setup(config: AnalysisConfig) -> dict[str, object]:
    """Return the settings that determine background sanity-check plots."""
    return {
        "energy_ranges_keV": [
            list(bounds) for bounds in config.background.energy_ranges_keV
        ],
        "bin_widths_keV": list(config.background.bin_widths_keV),
        "comparison_cut_profile": config.background.comparison_cut_profile,
    }


def manifest_stages(manifest: Mapping[str, object]) -> dict[str, object]:
    """Return the mutable stage mapping from a normalized manifest."""
    stages = manifest.get("stages")
    if not isinstance(stages, dict):
        msg = "Manifest stages must be a mutable dictionary"
        raise TypeError(msg)
    return stages


def stage_record(
    manifest: Mapping[str, object],
    stage: str,
) -> dict[str, object] | None:
    """Return one stage record when it is a mapping."""
    record = manifest_stages(manifest).get(stage)
    return dict(record) if isinstance(record, Mapping) else None


def mark_stages_stale(
    manifest: Mapping[str, object],
    stages: Sequence[str],
) -> None:
    """Mark existing downstream stage records stale without losing metadata."""
    records = manifest_stages(manifest)
    for stage in stages:
        record = records.get(stage)
        if isinstance(record, Mapping):
            records[stage] = {**record, "status": "stale"}

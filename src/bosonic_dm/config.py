# Copyright (C) 2025 Francesco Borra
#

"""Configuration layer for bosonic DM analysis."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ruamel.yaml import YAML


@dataclass(frozen=True)
class ProductionConfig:
    version: str
    reference_root: Path
    metadata_override: str | None


@dataclass(frozen=True)
class PathsConfig:
    simulation_root: Path
    data_root: Path
    parquet_root: Path
    dictionaries_root: Path
    inputs_root: Path
    plots_root: Path
    temporary_root: Path


@dataclass(frozen=True)
class FepWindowConfig:
    half_width_fwhm: float


@dataclass(frozen=True)
class InteractionConfig:
    name: str
    job_template: str
    make_lar_survival_plots: bool
    make_energy_spectra_plots: bool
    make_aoe_survival_plots: bool


@dataclass(frozen=True)
class BackgroundConfig:
    pet_glob: str
    comparison_cut_profile: str
    energy_ranges_keV: list[tuple[int, int]]
    bin_widths_keV: list[int]


@dataclass(frozen=True)
class OutputConfig:
    overwrite: bool
    save_plots: bool
    write_manifest: bool


@dataclass(frozen=True)
class AnalysisConfig:
    production: ProductionConfig
    paths: PathsConfig
    energies_keV: tuple[int, ...]
    fep_window: FepWindowConfig
    selections: tuple[str, ...]
    apply_lar_veto: bool
    interactions: Mapping[str, InteractionConfig]
    background: BackgroundConfig
    output: OutputConfig
    detector_groups: Path


def _resolve_path(path_str: str) -> Path:
    """Expand environment variables and return a Path object."""
    return Path(os.path.expandvars(path_str))


def load_analysis_config(path: str | Path) -> AnalysisConfig:
    """Load and validate the YAML configuration."""
    yaml = YAML(typ="safe")

    with Path(path).open("r") as f:
        raw_config = yaml.load(f)

    if raw_config.get("schema_version") != 1:
        msg = f"Unsupported schema version: {raw_config.get('schema_version')}"
        raise ValueError(msg)

    raw_prod = raw_config["production"]
    production = ProductionConfig(
        version=raw_prod["version"],
        reference_root=_resolve_path(raw_prod["reference_root"]),
        metadata_override=(
            os.path.expandvars(raw_prod["metadata_override"])
            if raw_prod.get("metadata_override")
            else None
        ),
    )

    raw_paths = raw_config["paths"]
    data_root_path = _resolve_path(raw_paths["data_root"])
    paths = PathsConfig(
        simulation_root=_resolve_path(raw_paths["simulation_root"]),
        data_root=data_root_path,
        parquet_root=data_root_path / "parquet",
        dictionaries_root=data_root_path / "dictionaries",
        inputs_root=_resolve_path(raw_paths.get("inputs_root", "data/inputs")),
        plots_root=_resolve_path(raw_paths.get("plots_root", "plots")),
        temporary_root=_resolve_path(raw_paths.get("temporary_root", "tmp")),
    )

    raw_analysis = raw_config["analysis"]
    energies_raw = raw_analysis["simulated_energies_keV"]
    if isinstance(energies_raw, dict):
        start = energies_raw["start"]
        stop = energies_raw["stop"]
        step = energies_raw.get("step", 10)
        energies_list = list(range(start, stop + 1, step))
    else:
        energies_list = energies_raw
    energies = tuple(sorted(set(energies_list)))
    if any(e <= 0 for e in energies):
        msg = "Energies must be positive."
        raise ValueError(msg)

    fwhm = raw_analysis["fep_window"]["half_width_fwhm"]
    if fwhm <= 0:
        msg = "half_width_fwhm must be > 0"
        raise ValueError(msg)
    fep_window = FepWindowConfig(half_width_fwhm=fwhm)

    selections = tuple(raw_analysis["selections"])

    raw_interactions = raw_config["interactions"]
    interactions = {}
    for name, data in raw_interactions.items():
        if "{energy}" not in data["job_template"]:
            msg = f"Interaction {name} job_template must contain '{{energy}}'"
            raise ValueError(msg)

        interactions[name] = InteractionConfig(
            name=name,
            job_template=data["job_template"],
            make_lar_survival_plots=data.get("make_lar_survival_plots", False),
            make_energy_spectra_plots=data.get("make_energy_spectra_plots", False),
            make_aoe_survival_plots=data.get("make_aoe_survival_plots", False),
        )

    raw_bg = raw_config["background"]
    if "apply_lar_veto" in raw_analysis:
        apply_lar_veto = raw_analysis["apply_lar_veto"]
    elif "apply_lar_veto" in raw_bg:
        logging.warning(
            "background.apply_lar_veto is deprecated; move it to "
            "analysis.apply_lar_veto."
        )
        apply_lar_veto = raw_bg["apply_lar_veto"]
    else:
        msg = "analysis.apply_lar_veto is required."
        raise ValueError(msg)
    if not isinstance(apply_lar_veto, bool):
        msg = "analysis.apply_lar_veto must be a boolean."
        raise ValueError(msg)

    background = BackgroundConfig(
        pet_glob=os.path.expandvars(raw_bg.get("pet_glob", "")),
        comparison_cut_profile=raw_bg["comparison_cut_profile"],
        energy_ranges_keV=[tuple(r) for r in raw_bg["energy_ranges_keV"]],
        bin_widths_keV=list(raw_bg["bin_widths_keV"]),
    )

    det_groups_str = raw_analysis.get(
        "detector_groups", "dictionaries/detector-grouping/groups_dict.yaml"
    )
    bg_det_groups = Path(det_groups_str)
    if not bg_det_groups.is_absolute():
        bg_det_groups = paths.inputs_root / bg_det_groups

    raw_out = raw_config["output"]
    output = OutputConfig(
        overwrite=raw_out.get("overwrite", False),
        save_plots=raw_out.get("save_plots", True),
        write_manifest=raw_out.get("write_manifest", True),
    )

    # Basic validations
    # Check if reference root exists, warning instead of crash, as user may not be on NERSC when testing locally
    if not production.reference_root.exists():
        logging.warning(
            f"Reference root {production.reference_root} does not exist. Validation skipped."
        )

    for p in [
        paths.data_root,
        paths.parquet_root,
        paths.dictionaries_root,
        paths.plots_root,
        paths.temporary_root,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    return AnalysisConfig(
        production=production,
        paths=paths,
        energies_keV=energies,
        fep_window=fep_window,
        selections=selections,
        apply_lar_veto=apply_lar_veto,
        interactions=interactions,
        background=background,
        output=output,
        detector_groups=bg_det_groups,
    )

# Copyright (C) 2026 Francesco Borra
#

"""Configuration layer for bosonic DM analysis."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ruamel.yaml import YAML

logger = logging.getLogger(__name__)


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


# Standard table of gamma-line centres masked in the background fit; selected
# with ``background.fit.peaks_keV: standard`` (the default).
DEFAULT_BACKGROUND_FIT_PEAKS_KEV = (
    238.632,  # Pb-212; merges 241.0 Ra-224, 242.0 Pb-214
    295.224,  # Pb-214
    351.932,  # Pb-214
    511.0,  # e+e- / Tl-208
    583.187,  # Tl-208
    609.312,  # Bi-214
    727.330,  # Bi-212
    860.557,  # Tl-208
    911.204,  # Ac-228
    968.971,  # Ac-228; merges 964.8
    1120.29,  # Bi-214
    1238.11,  # Bi-214
    1377.67,  # Bi-214
    1460.82,  # K-40
    1524.6,  # K-42; merges 1509.2, 1512.7
    1588.2,  # Ac-228; merges 1592.5 DEP
    1620.50,  # Bi-212; merges 1630.6
    1729.6,  # Bi-214
    1764.49,  # Bi-214
    1847.42,  # Bi-214
)
BACKGROUND_FIT_MODELS = ("no_linear", "full")
STANDARD_PEAKS_KEYWORD = "standard"


def resolve_background_fit_peaks(
    value: object,
    fit_range_keV: tuple[float, float],
    exclusion_width_keV: float,
) -> tuple[float, ...]:
    """Resolve ``background.fit.peaks_keV`` to the peaks masked in the fit.

    Parameters
    ----------
    value
        ``None`` or ``"standard"`` for the standard table
        :data:`DEFAULT_BACKGROUND_FIT_PEAKS_KEV`, or a sequence of line
        centres in keV. An empty sequence masks nothing.
    fit_range_keV
        Energy range of the fit.
    exclusion_width_keV
        Half-width of the mask around each centre.

    Returns
    -------
    tuple[float, ...]
        Sorted, unique centres whose mask reaches into the fit range
        (``low - width < centre < high + width``); the others are dropped.

    Raises
    ------
    ValueError
        If ``value`` is another string or holds a non-numeric entry.
    """
    if value is None or value == STANDARD_PEAKS_KEYWORD:
        centres: list[float] = list(DEFAULT_BACKGROUND_FIT_PEAKS_KEV)
        explicit = False
    elif isinstance(value, str) or not isinstance(value, Sequence):
        msg = (
            "background.fit.peaks_keV must be 'standard' or a list of line "
            f"centres in keV, got {value!r}"
        )
        raise ValueError(msg)
    else:
        centres = []
        for entry in value:
            if isinstance(entry, bool) or not isinstance(entry, int | float):
                msg = f"background.fit.peaks_keV entry {entry!r} is not a number"
                raise ValueError(msg)
            centres.append(float(entry))
        explicit = True

    low, high = fit_range_keV
    kept = tuple(
        sorted(
            {
                c
                for c in centres
                if low - exclusion_width_keV < c < high + exclusion_width_keV
            }
        )
    )
    dropped = sorted(set(centres) - set(kept))
    if dropped and explicit:
        logger.info(
            "background.fit.peaks_keV: ignoring %s keV, outside the fit range %s keV",
            dropped,
            list(fit_range_keV),
        )
    return kept


@dataclass(frozen=True)
class BackgroundFitSamplerConfig:
    """CmdStan sampler settings of the background fit."""

    chains: int = 4
    iter_warmup: int = 1000
    iter_sampling: int = 1000
    seed: int = 12345


@dataclass(frozen=True)
class BackgroundFitConfig:
    """Settings of the per-group Stan background fit.

    Attributes
    ----------
    spectrum_range_keV, spectrum_bin_width_keV
        Energy range and bin width of the prepared group spectrum.
    fit_range_keV
        Range of the spectrum passed to Stan.
    peaks_keV, peak_exclusion_width_keV
        Gamma-line centres masked in the likelihood, already restricted to
        the fit range (see :func:`resolve_background_fit_peaks`), and the
        half-width of the mask.
    prefit_dictionary, prefit_key
        YAML file with the per-group curve_fit prefits, and the entry used
        inside each group. A relative path is resolved against
        ``paths.inputs_root``.
    model
        ``"no_linear"`` fixes ``c0 = cQ = 0``; ``"full"`` samples them.
    shape_sd
        Log-space prior standard deviation of the Ar39 shape parameters.
    rate_prior_sd
        Log-space prior standard deviations of ``n2nu`` and ``ar39``.
    sampler
        CmdStan sampler settings.
    """

    spectrum_range_keV: tuple[float, float] = (60.0, 2000.0)
    spectrum_bin_width_keV: float = 1.0
    fit_range_keV: tuple[float, float] = (100.0, 1100.0)
    peaks_keV: tuple[float, ...] = DEFAULT_BACKGROUND_FIT_PEAKS_KEV
    peak_exclusion_width_keV: float = 5.0
    prefit_dictionary: Path = Path("dictionaries/prefit_dictionary.yaml")
    prefit_key: str = "fit_ar"
    model: str = "no_linear"
    shape_sd: float = 0.5
    rate_prior_sd: Mapping[str, float] = field(
        default_factory=lambda: {"n2nu": 0.5, "ar39": 2.0}
    )
    sampler: BackgroundFitSamplerConfig = field(
        default_factory=BackgroundFitSamplerConfig
    )

    def __post_init__(self) -> None:
        """Validate the model name."""
        if self.model not in BACKGROUND_FIT_MODELS:
            msg = (
                f"background.fit.model must be one of {BACKGROUND_FIT_MODELS}, "
                f"got {self.model!r}."
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class BackgroundConfig:
    pet_glob: str
    comparison_cut_profile: str
    energy_ranges_keV: list[tuple[int, int]]
    bin_widths_keV: list[int]
    fit: BackgroundFitConfig = field(default_factory=BackgroundFitConfig)


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
    # Run whose simulated efficiency replaces that of a run missing from the
    # simulation: {(period, run): (source_period, source_run)}.
    efficiency_run_substitutes: Mapping[tuple[str, str], tuple[str, str]] = field(
        default_factory=dict
    )


_PERIOD_RUN = re.compile(r"(p\d+)-(r\d+)")


def _period_run(text: object) -> tuple[str, str]:
    """Split a ``"p09-r004"`` label into ``("p09", "r004")``."""
    match = _PERIOD_RUN.fullmatch(str(text).strip())
    if match is None:
        msg = f"Expected a period-run label such as 'p09-r004', got {text!r}."
        raise ValueError(msg)
    return match.group(1), match.group(2)


def parse_run_substitutes(
    raw: Mapping[str, str] | None,
) -> dict[tuple[str, str], tuple[str, str]]:
    """Parse ``analysis.efficiency_run_substitutes``.

    Each entry maps a run without simulated primaries to the run whose
    efficiency it uses, e.g. ``{"p09-r004": "p09-r005"}``.

    Raises
    ------
    ValueError
        If a label is malformed, a run replaces itself, or a source run is
        itself replaced.
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        msg = "analysis.efficiency_run_substitutes must be a mapping."
        raise ValueError(msg)
    substitutes = {_period_run(key): _period_run(value) for key, value in raw.items()}
    for target, source in substitutes.items():
        if target == source:
            msg = f"Run {'-'.join(target)} cannot replace itself."
            raise ValueError(msg)
        if source in substitutes:
            msg = (
                f"Source run {'-'.join(source)} is itself replaced; "
                "point every run at a simulated run."
            )
            raise ValueError(msg)
    return dict(sorted(substitutes.items()))


def _resolve_path(path_str: str) -> Path:
    """Expand environment variables and return a Path object."""
    return Path(os.path.expandvars(path_str))


def parse_background_fit_config(
    raw: Mapping[str, object] | None,
    inputs_root: Path,
) -> BackgroundFitConfig:
    """Parse the optional ``background.fit`` section.

    Parameters
    ----------
    raw
        The ``background.fit`` mapping, or ``None`` when absent. Missing keys
        take the ``BackgroundFitConfig`` defaults.
    inputs_root
        Root for a relative ``prefit_dictionary``.

    Returns
    -------
    BackgroundFitConfig
        The validated settings.

    Raises
    ------
    ValueError
        If the section is not a mapping, has unknown keys, or names an unknown
        model.
    """
    defaults = BackgroundFitConfig()
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        msg = "background.fit must be a mapping."
        raise ValueError(msg)
    unknown = sorted(set(raw) - set(BackgroundFitConfig.__dataclass_fields__))
    if unknown:
        msg = f"Unknown background.fit keys: {unknown}"
        raise ValueError(msg)

    def _range(key: str, default: tuple[float, float]) -> tuple[float, float]:
        low, high = raw.get(key, default)
        return float(low), float(high)

    rate_prior_sd = dict(defaults.rate_prior_sd)
    raw_rate_sd = raw.get("rate_prior_sd") or {}
    unknown_rates = sorted(set(raw_rate_sd) - set(rate_prior_sd))
    if unknown_rates:
        msg = f"Unknown background.fit.rate_prior_sd keys: {unknown_rates}"
        raise ValueError(msg)
    rate_prior_sd.update({str(k): float(v) for k, v in raw_rate_sd.items()})

    raw_sampler = raw.get("sampler") or {}
    sampler_fields = set(BackgroundFitSamplerConfig.__dataclass_fields__)
    unknown_sampler = sorted(set(raw_sampler) - sampler_fields)
    if unknown_sampler:
        msg = f"Unknown background.fit.sampler keys: {unknown_sampler}"
        raise ValueError(msg)
    sampler = BackgroundFitSamplerConfig(
        **{key: int(value) for key, value in raw_sampler.items()}
    )

    fit_range_keV = _range("fit_range_keV", defaults.fit_range_keV)
    peak_exclusion_width_keV = float(
        raw.get("peak_exclusion_width_keV", defaults.peak_exclusion_width_keV)
    )

    prefit_dictionary = _resolve_path(
        str(raw.get("prefit_dictionary", defaults.prefit_dictionary))
    )
    if not prefit_dictionary.is_absolute():
        prefit_dictionary = inputs_root / prefit_dictionary

    return BackgroundFitConfig(
        spectrum_range_keV=_range("spectrum_range_keV", defaults.spectrum_range_keV),
        spectrum_bin_width_keV=float(
            raw.get("spectrum_bin_width_keV", defaults.spectrum_bin_width_keV)
        ),
        fit_range_keV=fit_range_keV,
        peaks_keV=resolve_background_fit_peaks(
            raw.get("peaks_keV"), fit_range_keV, peak_exclusion_width_keV
        ),
        peak_exclusion_width_keV=peak_exclusion_width_keV,
        prefit_dictionary=prefit_dictionary,
        prefit_key=str(raw.get("prefit_key", defaults.prefit_key)),
        model=str(raw.get("model", defaults.model)),
        shape_sd=float(raw.get("shape_sd", defaults.shape_sd)),
        rate_prior_sd=rate_prior_sd,
        sampler=sampler,
    )


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
        plots_root=data_root_path / "plots",
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
        fit=parse_background_fit_config(raw_bg.get("fit"), paths.inputs_root),
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
        efficiency_run_substitutes=parse_run_substitutes(
            raw_analysis.get("efficiency_run_substitutes")
        ),
    )

# Copyright (C) 2026 Francesco Borra
#

"""Background-analysis pipeline orchestration."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import dbetto
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from cmdstanpy import CmdStanModel
from matplotlib.figure import Figure

from bosonic_dm.background import (
    build_background_dataset,
    prepare_background_spectra,
    spectrum_in_range,
)
from bosonic_dm.config import AnalysisConfig
from bosonic_dm.cuts import compute_group_exposure
from bosonic_dm.fitting.background_fit import (
    NO_LINEAR,
    STAN_FILE,
    build_stan_data,
    inits_from_prefit,
    load_background_fit,
    priors_from_prefit,
    rate_prior_centres,
    run_background_fit,
    save_background_fit,
    summarize_background_fit,
)
from bosonic_dm.models import AnalysisArtifacts
from bosonic_dm.pipeline.context import build_analysis_context
from bosonic_dm.pipeline.manifest import (
    current_background_cut_setup,
    current_background_fit_setup,
    current_background_plot_setup,
    load_background_manifest,
    manifest_stages,
    mark_stages_stale,
    stage_record,
)
from bosonic_dm.plotting.background import (
    plot_background_partition_summary,
    plot_background_spectrum,
)
from bosonic_dm.plotting.background_fit import (
    plot_background_chains,
    plot_background_corner,
    plot_background_fit_residuals,
)
from bosonic_dm.yaml_io import write_yaml

logger = logging.getLogger(__name__)

BACKGROUND_STAGE_ORDER = ("build-dataset", "fit", "fit-plots")
BACKGROUND_DEFAULT_STAGES = BACKGROUND_STAGE_ORDER
BACKGROUND_STAGE_DEPENDENCIES = {
    "build-dataset": (),
    "fit": ("build-dataset",),
    "fit-plots": ("fit",),
}
# Parameters held fixed by build_stan_data for each background.fit.model.
BACKGROUND_FIT_FIXED = {"no_linear": NO_LINEAR, "full": None}
# Files written by save_background_fit in each group directory.
BACKGROUND_FIT_OUTPUT_FILES = (
    "posterior.nc",
    "bhat.parquet",
    "expected_counts_draws.npy",
    "summary.yaml",
)
BACKGROUND_FIT_PLOT_NAMES = ("residuals", "chains", "corner")
_CURRENT_STATUSES = frozenset({"completed", "cached"})


def resolve_background_stages(stages: Sequence[str]) -> tuple[str, ...]:
    """Expand requested background stages in execution order."""
    unknown = set(stages) - set(BACKGROUND_STAGE_ORDER)
    if unknown:
        msg = f"Unknown background stages: {sorted(unknown)}"
        raise ValueError(msg)

    resolved: set[str] = set()

    def add_with_dependencies(stage: str) -> None:
        for dependency in BACKGROUND_STAGE_DEPENDENCIES[stage]:
            add_with_dependencies(dependency)
        resolved.add(stage)

    for requested_stage in stages:
        add_with_dependencies(requested_stage)

    return tuple(stage for stage in BACKGROUND_STAGE_ORDER if stage in resolved)


def discover_background_inputs(pet_glob: str) -> tuple[Path, ...]:
    """Return PET files matching a filename glob in stable order."""
    glob_path = Path(pet_glob)
    return tuple(
        sorted(path for path in glob_path.parent.glob(glob_path.name) if path.is_file())
    )


def background_dataset_path(config: AnalysisConfig) -> Path:
    """Return the deterministic root of the background Parquet dataset."""
    return config.paths.parquet_root / "background"


def background_manifest_path(config: AnalysisConfig) -> Path:
    """Return the deterministic path of the background pipeline manifest."""
    return config.paths.data_root / "background_manifest.yaml"


def group_slug(group: str) -> str:
    """Return the filesystem name of a detector group (spaces become ``_``)."""
    return group.replace(" ", "_")


def background_fit_dir(config: AnalysisConfig, group: str) -> Path:
    """Return the output directory of one group's background fit."""
    return config.paths.data_root / "background_fits" / group_slug(group)


def background_fit_plot_paths(config: AnalysisConfig, group: str) -> dict[str, Path]:
    """Return the residual, chain and corner plot paths of one group fit."""
    plot_dir = config.paths.plots_root / "background_fit" / group_slug(group)
    return {name: plot_dir / f"{name}.png" for name in BACKGROUND_FIT_PLOT_NAMES}


def load_background_groups(config: AnalysisConfig) -> dbetto.AttrsDict:
    """Read the detector-group dictionary named by ``config.detector_groups``."""
    return dbetto.Props.read_from(str(config.detector_groups))


def resolve_background_groups(
    requested: Sequence[str] | None,
    available: Iterable[str],
) -> tuple[str, ...]:
    """Match requested group names against the detector-group dictionary.

    Parameters
    ----------
    requested
        Group names, each given exactly or with ``_`` in place of spaces
        (``"ICPC_group1"`` for ``"ICPC group1"``). ``None`` selects every
        available group.
    available
        Group names of the detector-group dictionary, in their stored order.

    Returns
    -------
    tuple[str, ...]
        Canonical group names, duplicates removed, in request order.

    Raises
    ------
    ValueError
        If a requested group is not available; the message lists the valid
        groups.
    """
    names = [str(name) for name in available]
    if requested is None:
        return tuple(names)
    resolved: list[str] = []
    for name in requested:
        if name in names:
            group = name
        elif name.replace("_", " ") in names:
            group = name.replace("_", " ")
        else:
            msg = f"Unknown background group {name!r}. Valid groups: {names}"
            raise ValueError(msg)
        if group not in resolved:
            resolved.append(group)
    return tuple(resolved)


def dataset_fingerprint(record: Mapping[str, Any], dataset_path: Path) -> str:
    """Return a short fingerprint of a background dataset from its manifest record.

    The fingerprint hashes the ``cut_setup`` and the sorted set of written and
    reused partitions (relative to ``dataset_path``) stored in the
    ``build-dataset`` record. No data file is read.
    """
    partitions = set()
    for key in ("written_partitions", "reused_partitions"):
        for path in record.get(key) or ():
            try:
                partitions.add(str(Path(path).relative_to(dataset_path)))
            except ValueError:
                partitions.add(str(path))
    payload = {
        "cut_setup": record.get("cut_setup"),
        "partitions": sorted(partitions),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def open_background_posterior(fit_dir: Path) -> Any:
    """Open the saved ``posterior.nc`` of a group fit as an xarray DataTree."""
    import xarray as xr  # noqa: PLC0415

    return xr.open_datatree(fit_dir / "posterior.nc")


def _resolve_prefit(
    prefit_dictionary: Mapping[str, Any],
    group: str,
    prefit_key: str,
) -> Mapping[str, Any]:
    """Return one group's prefit, accepting space- or ``_``-separated keys."""
    for candidate in (group, group.replace(" ", "_"), group.replace("_", " ")):
        if candidate in prefit_dictionary:
            return prefit_dictionary[candidate][prefit_key]
    msg = (
        f"No prefit for background group {group!r}. "
        f"Available prefits: {list(prefit_dictionary)}"
    )
    raise ValueError(msg)


def _group_records(
    stages_manifest: Mapping[str, Any],
    stage: str,
) -> dict[str, dict[str, Any]]:
    """Return the per-group records of a fit stage as plain dictionaries."""
    record = stages_manifest.get(stage)
    groups = record.get("groups") if isinstance(record, Mapping) else None
    if not isinstance(groups, Mapping):
        return {}
    return {
        str(name): dict(value)
        for name, value in groups.items()
        if isinstance(value, Mapping)
    }


def _set_group_records(
    stages_manifest: dict[str, Any],
    stage: str,
    records: Mapping[str, Mapping[str, Any]],
    status: str | None = None,
) -> None:
    """Store per-group records, keeping the other stage-level fields."""
    previous = stages_manifest.get(stage)
    base = dict(previous) if isinstance(previous, Mapping) else {}
    if status is not None:
        base["status"] = status
    base["groups"] = {name: dict(record) for name, record in records.items()}
    stages_manifest[stage] = base


def _mark_group_records_stale(
    manifest: Mapping[str, Any],
    stage: str,
    groups: Iterable[str] | None = None,
) -> None:
    """Mark the stage record and some (``None``: all) of its groups stale."""
    stages_manifest = manifest_stages(manifest)
    records = _group_records(stages_manifest, stage)
    selected = set(records) if groups is None else set(groups) & set(records)
    if not selected and groups is not None:
        return
    for name in selected:
        records[name]["status"] = "stale"
    if stage in stages_manifest:
        mark_stages_stale(manifest, (stage,))
        _set_group_records(stages_manifest, stage, records)


def _record_is_current(
    record: Mapping[str, Any] | None,
    *,
    fit_setup: Mapping[str, Any],
    fingerprint: str,
) -> bool:
    """Whether a per-group record matches the setup and its outputs exist."""
    if record is None or record.get("status") not in _CURRENT_STATUSES:
        return False
    if record.get("fit_setup") != fit_setup:
        return False
    if record.get("dataset_fingerprint") != fingerprint:
        return False
    outputs = record.get("outputs") or []
    return bool(outputs) and all(Path(path).exists() for path in outputs)


def _overall_status(statuses: Iterable[str]) -> str:
    """Combine per-group statuses into one stage status."""
    values = set(statuses)
    if not values:
        return "blocked"
    if values <= {"cached"}:
        return "cached"
    if values <= _CURRENT_STATUSES:
        return "completed"
    if values & _CURRENT_STATUSES:
        return "partial"
    return "blocked"


def _close(figure: object) -> None:
    """Close a Matplotlib figure returned by a plotting function."""
    if isinstance(figure, Figure):
        plt.close(figure)


def _fit_background_group(
    config: AnalysisConfig,
    group: str,
    spectrum: Mapping[str, Any],
    group_dict: Mapping[str, Any],
    eres_dict: Mapping[str, Any],
    prefit: Mapping[str, Any],
    model: CmdStanModel,
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Path], bool]:
    """Fit one detector group and save the posterior and its summary.

    Parameters
    ----------
    config
        Analysis configuration; ``background.fit`` sets priors and sampler.
    group
        Canonical detector-group name.
    spectrum
        The group's prepared spectrum, already cut to ``fit_range_keV``.
    group_dict, eres_dict
        Group selection and energy-resolution dictionaries for the exposure.
    prefit
        The group's curve_fit prefit (``prefit_key`` entry).
    model
        Compiled ``background.stan``.
    provenance
        Extra fields copied into the saved ``summary.yaml``.

    Returns
    -------
    tuple[dict[str, Path], bool]
        Paths written by ``save_background_fit`` and the diagnostics flag.
    """
    fit_config = config.background.fit
    fixed = BACKGROUND_FIT_FIXED[fit_config.model]
    exposure = compute_group_exposure(eres_dict, group_dict)
    n2nu, ar39 = rate_prior_centres(spectrum, exposure)

    priors = priors_from_prefit(prefit, shape_sd=fit_config.shape_sd)
    priors["log_rate_prior_mean"][:2] = np.log([n2nu, ar39])
    priors["log_rate_prior_sd"][:2] = [
        fit_config.rate_prior_sd["n2nu"],
        fit_config.rate_prior_sd["ar39"],
    ]
    data = build_stan_data(spectrum, priors, fixed=fixed)

    sampler = fit_config.sampler
    inits = inits_from_prefit(prefit, chains=sampler.chains, fixed=fixed)
    fit = run_background_fit(
        model,
        data,
        seed=sampler.seed,
        chains=sampler.chains,
        iter_warmup=sampler.iter_warmup,
        iter_sampling=sampler.iter_sampling,
        inits=inits,
    )
    summary = summarize_background_fit(fit, data)
    paths = save_background_fit(
        fit,
        summary,
        data,
        background_fit_dir(config, group),
        provenance={
            **provenance,
            "exposure_kg_yr": float(exposure),
            "rate_prior_centres": {"n2nu": float(n2nu), "ar39": float(ar39)},
        },
        stan_file=STAN_FILE,
    )
    return paths, bool(summary.passed)


def _plot_background_group(
    config: AnalysisConfig,
    group: str,
) -> list[Path]:
    """Draw the residual, chain and corner plots of one saved group fit.

    The plots are drawn from the saved artifacts only: the summary from
    ``load_background_fit`` and the full posterior from ``posterior.nc``.
    No refit is needed.
    """
    fit_dir = background_fit_dir(config, group)
    plot_paths = background_fit_plot_paths(config, group)
    plot_paths["residuals"].parent.mkdir(parents=True, exist_ok=True)
    summary = load_background_fit(fit_dir)
    posterior = open_background_posterior(fit_dir)
    try:
        _close(
            plot_background_fit_residuals(
                summary,
                summary.counts,
                title=group,
                posterior=posterior,
                output_path=plot_paths["residuals"],
            )
        )
        _close(
            plot_background_chains(
                posterior, title=group, output_path=plot_paths["chains"]
            )
        )
        _close(
            plot_background_corner(
                posterior, title=group, output_path=plot_paths["corner"]
            )
        )
    finally:
        close = getattr(posterior, "close", None)
        if callable(close):
            close()
    return [plot_paths[name] for name in BACKGROUND_FIT_PLOT_NAMES]


def _has_parquet_fragments(dataset_path: Path) -> bool:
    """Return whether a dataset directory contains at least one Parquet file."""
    first_fragment = next(dataset_path.rglob("*.parquet"), None)
    return dataset_path.is_dir() and first_fragment is not None


def _parquet_fragments(dataset_path: Path) -> tuple[Path, ...]:
    """Return existing background Parquet fragments in stable order."""
    if not dataset_path.is_dir():
        return ()
    return tuple(sorted(dataset_path.rglob("*.parquet")))


def _expected_sanity_plot_paths(config: AnalysisConfig) -> list[Path]:
    """Return the deterministic background sanity-check plot paths."""
    plot_dir = config.paths.plots_root / "background"
    paths = [
        plot_dir / f"spectrum_{emin}_{emax}keV_{bin_width}keV.png"
        for emin, emax in config.background.energy_ranges_keV
        for bin_width in config.background.bin_widths_keV
    ]
    paths.append(plot_dir / "partition_summary.png")
    return paths


def _warn(artifacts: AnalysisArtifacts, message: str, *args: object) -> None:
    """Log and retain a non-fatal pipeline warning."""
    rendered = message % args if args else message
    logger.warning(rendered)
    artifacts.warnings.append(rendered)


def _write_manifest(
    config: AnalysisConfig,
    manifest: dict[str, object],
    artifacts: AnalysisArtifacts,
) -> None:
    """Atomically write the cumulative background manifest."""
    if not config.output.write_manifest:
        return

    manifest_path = background_manifest_path(config)
    write_yaml(manifest, manifest_path)
    artifacts.manifest_path = manifest_path
    logger.info("Background pipeline manifest written to %s", manifest_path)


def run_background_analysis(
    config: AnalysisConfig,
    *,
    stages: Sequence[str] = BACKGROUND_DEFAULT_STAGES,
    groups: Sequence[str] | None = None,
    overwrite: bool | None = None,
) -> AnalysisArtifacts:
    """Build the background dataset, then fit and plot each detector group.

    Parameters
    ----------
    config
        Analysis configuration; ``background.fit`` holds the fit settings.
    stages
        Requested stages; dependencies are added (``fit-plots`` runs ``fit``
        and ``build-dataset`` too).
    groups
        Detector groups to fit and plot, given exactly or with ``_`` for
        spaces. ``None`` selects every group of the detector-group dictionary.
    overwrite
        Rebuild the dataset and refit and replot the selected groups even when
        their manifest records are current. ``None`` uses
        ``config.output.overwrite``.

    Returns
    -------
    AnalysisArtifacts
        Dataset, YAML and plot paths, per-stage status and warnings.

    Raises
    ------
    ValueError
        If a stage or a requested group is unknown.
    """
    artifacts = AnalysisArtifacts()
    requested_stages = tuple(stages)
    resolved_stages = resolve_background_stages(requested_stages)
    do_overwrite = overwrite if overwrite is not None else config.output.overwrite
    selected_groups: tuple[str, ...] = ()
    groups_dict: Mapping[str, Any] = {}
    if "fit" in resolved_stages:
        groups_dict = load_background_groups(config)
        selected_groups = resolve_background_groups(groups, groups_dict)

    logger.info(
        "─── background (stages: %s, overwrite=%s) ───",
        " → ".join(resolved_stages),
        do_overwrite,
    )
    manifest = load_background_manifest(background_manifest_path(config))
    manifest["data_root"] = str(config.paths.data_root)
    stages_manifest = manifest_stages(manifest)

    def save_manifest() -> None:
        _write_manifest(config, manifest, artifacts)

    cut_setup = current_background_cut_setup(config)
    plot_setup = current_background_plot_setup(config)
    prod = config.production
    base = prod.reference_root / prod.version
    config_json_path = base / "config.json"

    resolved_pet_glob = config.background.pet_glob

    if not resolved_pet_glob:
        try:
            prod_cfg = dbetto.Props.read_from(str(config_json_path), subst_pathvar=True)
            resolved_pet_glob = (
                f"{prod_cfg['setups']['l200']['paths']['tier_pet']}/phy/*.lh5"
            )
        except Exception as e:
            logger.error(f"Could not resolve tier_pet from {config_json_path}: {e}")
            msg = "No pet_glob provided in yaml and could not read config.json"
            raise ValueError(msg) from e

    pet_files = discover_background_inputs(resolved_pet_glob)
    dataset_path = background_dataset_path(config)
    written_partitions: tuple[Path, ...] = ()
    reused_partitions: tuple[Path, ...] = ()
    dataset_record = stage_record(manifest, "build-dataset")
    has_cached_dataset = _has_parquet_fragments(dataset_path)
    dataset_setup_matches = (
        dataset_record is not None and dataset_record.get("cut_setup") == cut_setup
    )
    dataset_recomputed = False
    dataset_is_current = False

    if "build-dataset" in resolved_stages:
        if has_cached_dataset and dataset_setup_matches and not do_overwrite:
            reused_partitions = _parquet_fragments(dataset_path)
            artifacts.dataset_paths.append(dataset_path)
            artifacts.stage_status["build-dataset"] = "cached"
            dataset_is_current = True
            logger.info("Stage build-dataset: cached ✓")
            stages_manifest["build-dataset"] = {
                "status": "cached",
                "outputs": [str(dataset_path)],
                "cut_setup": cut_setup,
                "pet_glob": resolved_pet_glob,
                "pet_files": [str(path) for path in pet_files],
                "written_partitions": [],
                "reused_partitions": [str(path) for path in reused_partitions],
            }
        elif not pet_files:
            _warn(
                artifacts,
                "Cannot build background dataset: no PET files match %s",
                resolved_pet_glob,
            )
            artifacts.stage_status["build-dataset"] = "blocked"
            blocked_record = dataset_record or {
                "outputs": [str(dataset_path)] if has_cached_dataset else []
            }
            stages_manifest["build-dataset"] = {
                **blocked_record,
                "status": "blocked",
                "pet_glob": resolved_pet_glob,
                "pet_files": [],
                "written_partitions": [],
                "reused_partitions": [],
            }
        else:
            context = build_analysis_context(config, load_eres=False)
            force_rebuild = do_overwrite or (
                has_cached_dataset and not dataset_setup_matches
            )
            build_result = build_background_dataset(
                pet_files,
                dataset_path,
                context,
                apply_lar_veto=config.apply_lar_veto,
                comparison_cut_profile=config.background.comparison_cut_profile,
                overwrite=force_rebuild,
            )
            written_partitions = build_result.written_paths
            reused_partitions = build_result.reused_paths
            if written_partitions or reused_partitions:
                artifacts.dataset_paths.append(dataset_path)
                dataset_is_current = True
            dataset_recomputed = bool(written_partitions)
            if written_partitions:
                dataset_status = "completed"
            elif reused_partitions:
                dataset_status = "cached"
            else:
                dataset_status = "blocked"
            artifacts.stage_status["build-dataset"] = dataset_status
            stages_manifest["build-dataset"] = {
                "status": dataset_status,
                "outputs": [str(dataset_path)] if dataset_is_current else [],
                "cut_setup": cut_setup,
                "pet_glob": resolved_pet_glob,
                "pet_files": [str(path) for path in pet_files],
                "written_partitions": [str(path) for path in written_partitions],
                "reused_partitions": [str(path) for path in reused_partitions],
            }
        current_dataset_record = stage_record(manifest, "build-dataset") or {}
        if dataset_is_current:
            stages_manifest["build-dataset"] = {
                **current_dataset_record,
                "dataset_fingerprint": dataset_fingerprint(
                    current_dataset_record, dataset_path
                ),
            }
        if dataset_recomputed:
            _mark_group_records_stale(manifest, "fit")
            _mark_group_records_stale(manifest, "fit-plots")
        save_manifest()

    # Sanity plots are automatic and deliberately not user-selectable.
    sanity_record = stage_record(manifest, "sanity-plots")
    expected_plot_paths = _expected_sanity_plot_paths(config)
    recorded_plot_paths = []
    if sanity_record is not None:
        outputs = sanity_record.get("outputs", [])
        if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
            recorded_plot_paths = [Path(path) for path in outputs]
    existing_plot_paths = list(
        dict.fromkeys(
            path
            for path in (*recorded_plot_paths, *expected_plot_paths)
            if path.exists()
        )
    )
    plot_setup_matches = (
        sanity_record is not None and sanity_record.get("plot_setup") == plot_setup
    )
    dataset_invalidated_plots = dataset_recomputed or (
        has_cached_dataset and not dataset_setup_matches
    )

    if config.output.save_plots and dataset_is_current:
        plot_dir = config.paths.plots_root / "background"
        for energy_range in config.background.energy_ranges_keV:
            for bin_width in config.background.bin_widths_keV:
                plot_path = plot_background_spectrum(
                    dataset_path,
                    energy_range_keV=tuple(energy_range),
                    bin_width_keV=bin_width,
                    comparison_cut_profile=config.background.comparison_cut_profile,
                    output_dir=plot_dir,
                )
                artifacts.plot_paths.append(plot_path)

        summary_path, plot_warnings = plot_background_partition_summary(
            dataset_path, plot_dir
        )
        artifacts.plot_paths.append(summary_path)
        for warning in plot_warnings:
            _warn(artifacts, warning)
        stages_manifest["sanity-plots"] = {
            "enabled": True,
            "status": "completed",
            "outputs": [str(path) for path in artifacts.plot_paths],
            "plot_setup": plot_setup,
        }
    elif config.output.save_plots:
        blocked_sanity_record = sanity_record or {
            "outputs": [str(path) for path in existing_plot_paths]
        }
        stages_manifest["sanity-plots"] = {
            **blocked_sanity_record,
            "enabled": True,
            "status": "blocked",
        }
    elif not existing_plot_paths:
        stages_manifest["sanity-plots"] = {
            "enabled": False,
            "status": "disabled",
            "outputs": [],
            "plot_setup": plot_setup,
        }
    elif dataset_invalidated_plots or not plot_setup_matches:
        stages_manifest["sanity-plots"] = {
            "enabled": False,
            "status": "stale",
            "outputs": [str(path) for path in existing_plot_paths],
            "plot_setup": (sanity_record.get("plot_setup") if sanity_record else None),
        }
    else:
        recorded_status = sanity_record.get("status") if sanity_record else None
        preserved_status = (
            recorded_status
            if recorded_status in {"completed", "cached", "partial"}
            else "stale"
        )
        stages_manifest["sanity-plots"] = {
            **(sanity_record or {}),
            "enabled": False,
            "status": preserved_status,
            "outputs": [str(path) for path in existing_plot_paths],
            "plot_setup": plot_setup,
        }

    fingerprint = (
        (stage_record(manifest, "build-dataset") or {}).get("dataset_fingerprint")
        if dataset_is_current
        else None
    )
    fit_setup = current_background_fit_setup(config)
    fit_statuses: dict[str, str] = {}

    if "fit" in resolved_stages:
        fit_statuses = _run_fit_stage(
            config,
            manifest,
            artifacts,
            selected_groups=selected_groups,
            groups_dict=groups_dict,
            dataset_path=dataset_path,
            fingerprint=fingerprint,
            fit_setup=fit_setup,
            overwrite=do_overwrite,
            save_manifest=save_manifest,
        )

    if "fit-plots" in resolved_stages:
        _run_fit_plots_stage(
            config,
            manifest,
            artifacts,
            selected_groups=selected_groups,
            fit_statuses=fit_statuses,
            fingerprint=fingerprint,
            fit_setup=fit_setup,
            overwrite=do_overwrite,
            save_manifest=save_manifest,
        )

    manifest["last_run"] = {
        "requested_stages": list(requested_stages),
        "resolved_stages": list(resolved_stages),
        "groups": list(selected_groups),
        "overwrite": do_overwrite,
        "warnings": list(artifacts.warnings),
    }
    _write_manifest(config, manifest, artifacts)
    return artifacts


def _run_fit_stage(
    config: AnalysisConfig,
    manifest: dict[str, Any],
    artifacts: AnalysisArtifacts,
    *,
    selected_groups: Sequence[str],
    groups_dict: Mapping[str, Any],
    dataset_path: Path,
    fingerprint: str | None,
    fit_setup: Mapping[str, Any],
    overwrite: bool,
    save_manifest: Callable[[], None],
) -> dict[str, str]:
    """Fit or reuse each selected group; return the per-group statuses."""
    stages_manifest = manifest_stages(manifest)
    records = _group_records(stages_manifest, "fit")
    if fingerprint is None:
        _warn(artifacts, "Cannot fit background: the dataset is not available")
        artifacts.stage_status["fit"] = "blocked"
        _set_group_records(stages_manifest, "fit", records, status="blocked")
        save_manifest()
        return dict.fromkeys(selected_groups, "blocked")

    statuses: dict[str, str] = {}
    to_fit: list[str] = []
    for group in selected_groups:
        record = records.get(group)
        if not overwrite and _record_is_current(
            record, fit_setup=fit_setup, fingerprint=fingerprint
        ):
            statuses[group] = "cached"
            records[group] = {**(record or {}), "status": "cached"}
            artifacts.yaml_paths.append(
                background_fit_dir(config, group) / "summary.yaml"
            )
            logger.info("Stage fit [%s]: cached ✓", group)
        else:
            to_fit.append(group)

    if to_fit:
        fit_config = config.background.fit
        eres_dict = dbetto.Props.read_from(
            str(config.paths.inputs_root / "dictionaries" / "eres_dict.yaml")
        )
        # Already resolved against inputs_root in parse_background_fit_config.
        prefit_dictionary = dbetto.Props.read_from(str(fit_config.prefit_dictionary))
        prefits = {
            group: _resolve_prefit(prefit_dictionary, group, fit_config.prefit_key)
            for group in to_fit
        }
        spectra = prepare_background_spectra(
            pl.scan_parquet(dataset_path, hive_partitioning=True),
            {group: groups_dict[group] for group in to_fit},
            eres_dict,
            energy_range_keV=fit_config.spectrum_range_keV,
            bin_width_keV=fit_config.spectrum_bin_width_keV,
            peak_centres_keV=list(fit_config.peaks_keV),
            exclusion_width_keV=fit_config.peak_exclusion_width_keV,
        )
        model = CmdStanModel(stan_file=str(STAN_FILE))
        for group in to_fit:
            logger.info("Stage fit [%s]: sampling", group)
            spectrum = spectrum_in_range(spectra[group], fit_config.fit_range_keV)
            paths, passed = _fit_background_group(
                config,
                group,
                spectrum,
                groups_dict[group],
                eres_dict,
                prefits[group],
                model,
                provenance={
                    "group": group,
                    "dataset": str(dataset_path),
                    "dataset_fingerprint": fingerprint,
                    "production_version": config.production.version,
                    "model": fit_config.model,
                    "shape_sd": fit_config.shape_sd,
                    "peaks_keV": list(fit_config.peaks_keV),
                    "exclusion_width_keV": fit_config.peak_exclusion_width_keV,
                    "fit_setup": dict(fit_setup),
                },
            )
            if not passed:
                _warn(
                    artifacts,
                    "Background fit for %s did not pass the sampler diagnostics",
                    group,
                )
            statuses[group] = "completed"
            records[group] = {
                "status": "completed",
                "fit_setup": dict(fit_setup),
                "dataset_fingerprint": fingerprint,
                "outputs": [str(paths[key]) for key in sorted(paths)],
                "passed_diagnostics": passed,
            }
            artifacts.yaml_paths.append(paths["summary"])
            _set_group_records(
                stages_manifest,
                "fit",
                records,
                status=_overall_status(statuses.values()),
            )
            _mark_group_records_stale(manifest, "fit-plots", (group,))
            save_manifest()

    status = _overall_status(statuses.values())
    artifacts.stage_status["fit"] = status
    _set_group_records(stages_manifest, "fit", records, status=status)
    save_manifest()
    return statuses


def _run_fit_plots_stage(
    config: AnalysisConfig,
    manifest: dict[str, Any],
    artifacts: AnalysisArtifacts,
    *,
    selected_groups: Sequence[str],
    fit_statuses: Mapping[str, str],
    fingerprint: str | None,
    fit_setup: Mapping[str, Any],
    overwrite: bool,
    save_manifest: Callable[[], None],
) -> None:
    """Draw or reuse the fit plots of each selected group."""
    stages_manifest = manifest_stages(manifest)
    records = _group_records(stages_manifest, "fit-plots")
    if not config.output.save_plots:
        artifacts.stage_status["fit-plots"] = "disabled"
        _set_group_records(stages_manifest, "fit-plots", records, status="disabled")
        save_manifest()
        return

    statuses: dict[str, str] = {}
    for group in selected_groups:
        if fingerprint is None or fit_statuses.get(group) not in _CURRENT_STATUSES:
            _warn(artifacts, "Cannot plot background fit for %s: no current fit", group)
            statuses[group] = "blocked"
            continue
        record = records.get(group)
        if not overwrite and _record_is_current(
            record, fit_setup=fit_setup, fingerprint=fingerprint
        ):
            statuses[group] = "cached"
            records[group] = {**(record or {}), "status": "cached"}
            artifacts.plot_paths.extend(
                background_fit_plot_paths(config, group)[name]
                for name in BACKGROUND_FIT_PLOT_NAMES
            )
            logger.info("Stage fit-plots [%s]: cached ✓", group)
            continue
        plot_paths = _plot_background_group(config, group)
        artifacts.plot_paths.extend(plot_paths)
        statuses[group] = "completed"
        records[group] = {
            "status": "completed",
            "fit_setup": dict(fit_setup),
            "dataset_fingerprint": fingerprint,
            "outputs": [str(path) for path in plot_paths],
        }
        _set_group_records(
            stages_manifest,
            "fit-plots",
            records,
            status=_overall_status(statuses.values()),
        )
        save_manifest()

    status = _overall_status(statuses.values())
    artifacts.stage_status["fit-plots"] = status
    _set_group_records(stages_manifest, "fit-plots", records, status=status)
    save_manifest()

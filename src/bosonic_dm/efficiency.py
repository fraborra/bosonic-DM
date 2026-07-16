# Copyright (C) 2025 Francesco Borra
#

"""Detection efficiency computation for bosonic-DM analysis."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import awkward as ak
import numpy as np
import polars as pl
from lgdo import lh5
from tqdm.notebook import tqdm

from bosonic_dm.stats import bayesian_efficiency

logger = logging.getLogger(__name__)


def compute_efficiency_from_lazyframe(
    lf: pl.LazyFrame,
    eres_dict: dict,
    simulated_energies: Sequence[int],
    chmap: object,
    scratch_folder: str | Path = "",
    job_base: str = "",
    *,
    single_site: bool | None = None,
    has_aoe: bool | None = None,
    vertex_counts: Mapping[int, Mapping[str, int]] | None = None,
) -> dict:
    """Compute per-detector efficiencies from a Polars lazy scan.

    For every combination of simulated energy and detector listed in
    *eres_dict*, the function:

    1. Looks up the detector-specific FWHM from *eres_dict*.
    2. Defines the full-energy-peak integration window as
       ``[e_value - 2·FWHM, e_value + 2·FWHM]``, where ``e_value`` is the
       reconstructed energy corresponding to the simulated energy ``ene``.
    3. Collects the lazy frame *lf* **once per energy** and uses a
       vectorised join + group-by to count events inside the FEP window
       for every detector simultaneously.
    4. Reads ``n_primaries`` from the STP files for each detector.
    5. Computes ``ratio = n_events / n_primaries``
       (or ``nan`` when ``n_primaries == 0``).

    Parameters
    ----------
    lf
        A Polars *lazy* scan of the parquet dataset partitioned by ``sim_e``.
        Expected columns: ``rawid``, ``energy``, ``sim_e``, ``is_good_channel``.
    eres_dict
        Nested dictionary with structure
        ``{energy: {detector_name: {"fwhm": float, ...}, ...}, ...}``
        as produced by the resolution-extraction pipeline
        (e.g. ``eres_per_det_tot.yaml``).
    simulated_energies
        List of simulated energies (in keV) to iterate over. The ``ene``
        in the loop corresponds to that simulated-energy partition.
    chmap
        LEGEND channel-map object (``LegendMetadata.channelmap(...)``).
        Must support ``chmap[det_name].daq.rawid``.
    scratch_folder
        Base scratch directory containing the generated LH5 STP files.
    job_base
        Job string template containing the ``{ene}`` placeholder.
    single_site
        If True, keep only rows where ``has_aoe`` and ``is_single_site`` are True.
        If False, keep only rows where ``has_aoe`` is True and ``is_single_site`` is False.
        If None (default), no filtering is applied.
    has_aoe
        If True, keep only rows where ``has_aoe`` is True.
        If None (default), no filtering is applied. Automatically set to True if ``single_site`` is used.
    vertex_counts
        Optional pre-computed vertex counts, structured as
        ``{energy: {detector_name: n_vertices, ...}, ...}``.
        When provided for a given energy, the function uses these counts
        as ``n_primaries`` instead of reading STP files.  Produced by
        :func:`~bosonic_dm.geometry.aggregate_vertex_counts`.

    Returns
    -------
    dict
        Nested dictionary with structure::

            {
                energy: {
                    detector_name: {
                        "n_events":         int,
                        "n_primaries":      int,
                        "ratio":            float,
                        "ratio_sigma":      float,
                        "ratio_sigma_freq": float,
                        "ratio_syst_fwhm":  float,
                        "expo":             float,   # from eres_dict
                    },
                    ...
                },
                ...
            }
    """
    if single_site is not None:
        has_aoe = True

    ratio_dict: dict = {}

    for ene in tqdm(simulated_energies):
        ene_key = int(ene)
        ratio_dict[ene_key] = {}

        # Skip energies not present in the resolution dictionary
        if ene_key not in eres_dict:
            logger.warning("Energy %d keV not found in eres_dict, skipping", ene_key)
            continue

        det_eres = eres_dict[ene_key]

        # Determine whether we have pre-computed vertex counts for this energy.
        vtx_counts_for_ene = (
            vertex_counts.get(ene_key) if vertex_counts is not None else None
        )

        # Fall back to STP reading only when vertex counts are unavailable.
        stp_files: list[str] = []
        if vtx_counts_for_ene is None:
            job_string = job_base.format(ene=ene)
            stp_files = [
                str(p)
                for p in Path(
                    f"{scratch_folder}/generated/tier/stp/{job_string}/"
                ).glob(f"l200cfg01-{job_string}-job_*-tier_stp.lh5")
            ]

        # --- Phase 1: build per-detector info and read n_primaries ----------
        det_rows: list[dict] = []
        n_prim_map: dict[str, int] = {}
        expo_map: dict[str, float] = {}

        for det_name, eres_info in det_eres.items():
            try:
                rawid = chmap[det_name].daq.rawid
            except (KeyError, AttributeError):
                logger.warning("Cannot resolve rawid for %s, skipping", det_name)
                continue

            # Use pre-computed vertex counts when available.
            if vtx_counts_for_ene is not None:
                n_primaries = vtx_counts_for_ene.get(det_name, 0)
            else:
                n_primaries = 0
                for stp_file in stp_files:
                    stp_ge = lh5.read_as(f"/stp/{det_name}", stp_file, library="ak")
                    n_primaries += len(np.unique(ak.to_numpy(stp_ge.evtid)))
            n_prim_map[det_name] = n_primaries
            expo_map[det_name] = float(eres_info.get("expo", 0.0))

            fwhm = float(eres_info["fwhm"])
            fwhm_unc = float(eres_info.get("unc", 0.0))
            fwhm_up = fwhm + fwhm_unc
            fwhm_down = max(fwhm - fwhm_unc, 0.0)

            det_rows.append(
                {
                    "rawid": rawid,
                    "det_name": det_name,
                    "low": ene - 2.0 * fwhm,
                    "high": ene + 2.0 * fwhm,
                    "low_up": ene - 2.0 * fwhm_up,
                    "high_up": ene + 2.0 * fwhm_up,
                    "low_down": ene - 2.0 * fwhm_down,
                    "high_down": ene + 2.0 * fwhm_down,
                }
            )

        if not det_rows:
            continue

        det_df = pl.DataFrame(det_rows)
        known_rawids = det_df["rawid"].to_list()

        # --- Phase 2: single collect per energy -----------------------------
        filter_expr = (
            (pl.col("is_good_channel"))
            & (pl.col("sim_e") == ene)
            & (pl.col("rawid").is_in(known_rawids))
        )
        if has_aoe is True:
            filter_expr &= pl.col("has_aoe")

        if single_site is True:
            filter_expr &= pl.col("is_single_site")
        elif single_site is False:
            filter_expr &= ~pl.col("is_single_site")

        df_ene = lf.filter(filter_expr).select("rawid", "energy").collect()

        # --- Phase 3: vectorised FEP counting via join + group_by -----------
        df_joined = df_ene.join(det_df, on="rawid")

        counts = df_joined.group_by("det_name").agg(
            pl.col("energy")
            .is_between(pl.col("low"), pl.col("high"))
            .sum()
            .alias("n_events"),
            pl.col("energy")
            .is_between(pl.col("low_up"), pl.col("high_up"))
            .sum()
            .alias("n_events_up"),
            pl.col("energy")
            .is_between(pl.col("low_down"), pl.col("high_down"))
            .sum()
            .alias("n_events_down"),
        )

        # Index counts by detector name for fast lookup
        counts_map = {row["det_name"]: row for row in counts.iter_rows(named=True)}

        # --- Phase 4: compute efficiencies ----------------------------------
        for det_name, n_primaries in n_prim_map.items():
            row = counts_map.get(det_name)

            # When has_aoe filtering is active, detectors with no surviving
            # events likely lack AoE information entirely. We skip saving them.
            if has_aoe is True and row is None:
                continue

            n_events = row["n_events"] if row else 0
            n_events_up = row["n_events_up"] if row else 0
            n_events_down = row["n_events_down"] if row else 0

            if n_primaries > 0:
                eff = n_events / n_primaries
                ratio_sigma_freq = float(np.sqrt(eff * (1.0 - eff) / n_primaries))
            else:
                ratio_sigma_freq = 0.0

            ratio, ratio_sigma = bayesian_efficiency(n_events, n_primaries)
            ratio_up, _ = bayesian_efficiency(n_events_up, n_primaries)
            ratio_down, _ = bayesian_efficiency(n_events_down, n_primaries)

            ratio_syst_fwhm = abs(ratio_up - ratio_down) / 2.0

            ratio_dict[ene_key][det_name] = {
                "n_events": n_events,
                "n_primaries": n_primaries,
                "ratio": ratio,
                "ratio_sigma": ratio_sigma,
                "ratio_sigma_freq": ratio_sigma_freq,
                "ratio_syst_fwhm": ratio_syst_fwhm,
                "expo": expo_map[det_name],
            }

    return ratio_dict


def filter_non_zero_efficiency(ratio_dict: Mapping) -> dict:
    """Filter ratio_dict to keep only entries where ratio (efficiency) is not zero.

    Parameters
    ----------
    ratio_dict
        Nested dictionary of efficiencies, structured as:
        ``{energy: {detector_name: {"ratio": float, ...}, ...}, ...}``.

    Returns
    -------
    dict
        A new nested dictionary containing only the detectors (and energies)
        with a non-zero ratio.
    """
    filtered: dict = {}
    for ene, det_dict in ratio_dict.items():
        filtered_dets = {}
        for det_name, info in det_dict.items():
            if info.get("ratio", 0.0) != 0.0:
                filtered_dets[det_name] = info
        if filtered_dets:
            filtered[ene] = filtered_dets
    return filtered

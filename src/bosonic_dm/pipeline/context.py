# Copyright (C) 2025 Francesco Borra
#

"""Shared analysis context initialization."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import dbetto
from legendmeta import LegendMetadata

from bosonic_dm.config import AnalysisConfig
from bosonic_dm.resolution import get_channelmap_cached


@dataclass
class AnalysisContext:
    config: AnalysisConfig
    timestamp: str
    lmeta: LegendMetadata
    eres_dict: dict[str, Any]
    _run_start_keys: dict[tuple[str, str], str] = field(
        default_factory=dict, init=False, repr=False
    )

    def get_channelmap(self, on: str) -> Any:
        """Get the channel map for a given timestamp, caching it to avoid repeated filesystem reads."""
        return get_channelmap_cached(self.lmeta, on)

    def get_run_start_key(self, period: str, run: str) -> str:
        """Return and cache the physics start key for one period and run."""
        cache_key = (period, run)
        if cache_key in self._run_start_keys:
            return self._run_start_keys[cache_key]

        try:
            start_key = self.lmeta.dataprod.runinfo[period][run].phy.start_key
        except (KeyError, TypeError, AttributeError) as exc:
            msg = f"Could not determine physics start_key for {period}-{run}."
            raise ValueError(msg) from exc

        self._run_start_keys[cache_key] = start_key
        return start_key

    def get_channelmap_for_run(self, period: str, run: str) -> Any:
        """Return the cached channel map valid for one period and run."""
        return self.get_channelmap(self.get_run_start_key(period, run))

    def get_channelmap_simulation(self) -> Any:
        """Return a representative channelmap for simulation by using the first available physics run."""
        # TODO: check this logic tomorrow
        try:
            runinfo = self.lmeta.dataprod.runinfo
            first_period = next(iter(runinfo))
            first_run = next(iter(runinfo[first_period]))
            start_key = runinfo[first_period][first_run].phy.start_key
        except (StopIteration, KeyError, TypeError, AttributeError) as e:
            msg = "Could not determine a representative start_key for simulation from runinfo."
            raise ValueError(msg) from e
        return self.get_channelmap(start_key)


logger = logging.getLogger(__name__)


def build_analysis_context(
    config: AnalysisConfig,
    *,
    load_eres: bool = True,
) -> AnalysisContext:
    """Initialize metadata and channel map once to avoid repeated queries."""
    timestamp = datetime.now().isoformat(timespec="seconds")
    prod = config.production
    base = prod.reference_root / prod.version
    config_json_path = base / "config.json"

    prod_cfg = dbetto.Props.read_from(str(config_json_path), subst_pathvar=True)
    metadata_path = prod_cfg["setups"]["l200"]["paths"]["metadata"]

    if prod.metadata_override:
        metadata_path = prod.metadata_override

    logger.info(
        "Initializing LegendMetadata from %s (this may take up to a minute)...",
        metadata_path,
    )
    lmeta = LegendMetadata(path=metadata_path)
    eres_dict: dict[str, Any] = {}
    if load_eres:
        eres_dict_path = config.paths.inputs_root / "dictionaries" / "eres_dict.yaml"
        eres_dict = dbetto.Props.read_from(str(eres_dict_path))

    return AnalysisContext(
        config=config,
        timestamp=timestamp,
        lmeta=lmeta,
        eres_dict=eres_dict,
    )

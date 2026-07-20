# Copyright (C) 2025 Francesco Borra
#

"""Shared analysis context initialization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
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

    def get_channelmap(self, on: str) -> Any:
        """Get the channel map for a given timestamp, caching it to avoid repeated filesystem reads."""
        return get_channelmap_cached(self.lmeta, on)

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


def build_analysis_context(config: AnalysisConfig) -> AnalysisContext:
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
    # Load eres_dict
    # Usually it's located in the dictionaries_root configured by the user
    eres_dict_path = (
        config.paths.calibration_dictionaries_root / "eres_per_det_tot.yaml"
    )
    eres_dict = dbetto.Props.read_from(str(eres_dict_path))

    return AnalysisContext(
        config=config,
        timestamp=timestamp,
        lmeta=lmeta,
        eres_dict=eres_dict,
    )

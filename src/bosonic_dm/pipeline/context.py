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


@dataclass
class AnalysisContext:
    config: AnalysisConfig
    timestamp: str
    lmeta: LegendMetadata
    eres_dict: dict[str, Any]
    _chmap_cache: dict[str, Any]

    def get_channelmap(
        self, period: str | None = None, *, on: str | None = None
    ) -> Any:
        """Get the channel map for a given period or timestamp, caching it to avoid repeated filesystem reads."""
        cache_key = on if on is not None else period
        if not cache_key:
            msg = "Must provide either 'period' or 'on'."
            raise ValueError(msg)

        if cache_key not in self._chmap_cache:
            if on is not None:
                self._chmap_cache[cache_key] = self.lmeta.channelmap(on=on)
            else:
                self._chmap_cache[cache_key] = self.lmeta.channelmap(period)
        return self._chmap_cache[cache_key]

    def get_channelmap_simulation(self) -> Any:
        """Return a representative channelmap for simulation (typically p03)."""
        return self.get_channelmap("p03")


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
        _chmap_cache={},
    )

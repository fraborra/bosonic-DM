# Copyright (C) 2026 Francesco Borra
#

"""Inputs of the signal fit and of the expected-sensitivity study.

Everything is read from the pipeline outputs named by an analysis config: the
background fit of every group, the detector groups, the resolution dictionary
and the signal efficiency table.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dbetto import AttrsDict, Props

from bosonic_dm.config import AnalysisConfig
from bosonic_dm.efficiency import efficiency_table_path, load_efficiency_table
from bosonic_dm.fitting.signal_template import (
    BackgroundPrediction,
    GroupEfficiency,
    group_efficiency,
    load_background_prediction,
)
from bosonic_dm.pipeline.background import background_fit_dir, load_background_groups

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalInputs:
    """Per-group inputs of ``build_signal_window``, read from the pipeline."""

    backgrounds: dict[str, BackgroundPrediction]
    groups_dict: dict[str, Mapping]
    eres_dict: AttrsDict
    group_efficiencies: dict[str, GroupEfficiency]
    fit_dirs: dict[str, Path]
    efficiency_table: Path

    @property
    def window_options(self) -> dict[str, object]:
        """Keyword arguments of ``build_signal_window`` and ``expected_limits``."""
        return {
            "eres_dict": self.eres_dict,
            "groups_dict": self.groups_dict,
            "group_efficiencies": self.group_efficiencies,
        }


def load_signal_inputs(
    config: AnalysisConfig,
    *,
    groups: Sequence[str] | None = None,
    interaction: str = "axio-electric",
    selection: str = "all",
) -> SignalInputs:
    """Read the background fits, groups, resolution and efficiencies.

    Parameters
    ----------
    config
        Analysis configuration.
    groups
        Groups to use; default all groups in ``config.detector_groups``.
    interaction
        Interaction of the efficiency table, e.g. ``"axio-electric"``.
    selection
        Efficiency selection.

    Returns
    -------
    SignalInputs
        The inputs, with the groups in the requested order.

    Raises
    ------
    ValueError
        If a requested group is not in ``config.detector_groups``.
    """
    groups_dict = load_background_groups(config)
    names = list(groups) if groups else list(groups_dict)
    missing = sorted(set(names) - set(groups_dict))
    if missing:
        msg = f"Groups not in {config.detector_groups}: {missing}."
        raise ValueError(msg)
    groups_dict = {name: groups_dict[name] for name in names}

    fit_dirs = {name: background_fit_dir(config, name) for name in names}
    backgrounds = {
        name: load_background_prediction(fit_dirs[name], group=name) for name in names
    }
    failed = [name for name, b in backgrounds.items() if not b.passed_diagnostics]
    if failed:
        logger.warning("Background fits that failed their diagnostics: %s", failed)

    eres_path = config.paths.inputs_root / "dictionaries" / "eres_dict.yaml"
    eres_dict = Props.read_from(str(eres_path))
    table_path = efficiency_table_path(config.paths.dictionaries_root, interaction)
    efficiencies = group_efficiency(
        load_efficiency_table(table_path), groups_dict, selection=selection
    )
    return SignalInputs(
        backgrounds=backgrounds,
        groups_dict=groups_dict,
        eres_dict=eres_dict,
        group_efficiencies=efficiencies,
        fit_dirs=fit_dirs,
        efficiency_table=table_path,
    )

# Copyright (C) 2025 Francesco Borra
#

"""Data structures and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AnalysisArtifacts:
    dataset_paths: list[Path] = field(default_factory=list)
    yaml_paths: list[Path] = field(default_factory=list)
    plot_paths: list[Path] = field(default_factory=list)
    manifest_path: Path | None = None
    stage_status: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PrimaryCounts:
    counts: dict[int, dict[str, int]]

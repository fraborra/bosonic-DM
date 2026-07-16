"""Data structures and result types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalysisArtifacts:
    dataset_paths: list[Path]
    yaml_paths: list[Path]
    plot_paths: list[Path]
    manifest_path: Path | None = None


@dataclass(frozen=True)
class PrimaryCounts:
    counts: dict[int, dict[str, int]]

"""Background-analysis pipeline."""

from __future__ import annotations

from collections.abc import Sequence

from bosonic_dm.config import AnalysisConfig
from bosonic_dm.models import AnalysisArtifacts


def run_background_analysis(
    config: AnalysisConfig,  # noqa: ARG001
    *,
    stages: Sequence[str] = ("build-dataset", "summaries", "plots"),  # noqa: ARG001
    overwrite: bool | None = None,  # noqa: ARG001
) -> AnalysisArtifacts:
    """Run the background analysis pipeline."""
    # Step 1: Create shared context
    # from bosonic_dm.pipeline.context import build_analysis_context
    # context = build_analysis_context(config)

    # Step 2: Extract cut profiles
    # Step 3: Build background datasets
    # Step 4: Compute summaries
    # Step 5: Write YAML products
    # Step 6: Make plots

    return AnalysisArtifacts(
        dataset_paths=[],
        yaml_paths=[],
        plot_paths=[],
    )

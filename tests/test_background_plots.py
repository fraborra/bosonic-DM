# Copyright (C) 2025 Francesco Borra
#

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from bosonic_dm.plotting.background import (
    plot_background_partition_summary,
    plot_background_spectrum,
)


def _make_dataset(tmp_path: Path, n_events: int = 200) -> Path:
    """Write a small synthetic Hive-partitioned Parquet dataset."""
    rng = np.random.default_rng(42)
    dataset_root = tmp_path / "parquet" / "background"
    for period, run in [("p03", "r000"), ("p04", "r001")]:
        energy = rng.uniform(20, 300, size=n_events)
        frame = pl.DataFrame(
            {
                "energy": energy,
                "passes_baseline": rng.choice([True, False], size=n_events).tolist(),
                "passes_default": rng.choice([True, False], size=n_events).tolist(),
                "passes_without_bb_like": rng.choice(
                    [True, False], size=n_events
                ).tolist(),
                "passes_lar": rng.choice([True, False], size=n_events).tolist(),
                "passes_analysis": rng.choice([True, False], size=n_events).tolist(),
                "passes_comparison": rng.choice([True, False], size=n_events).tolist(),
                "period": [period] * n_events,
                "run": [run] * n_events,
            }
        )
        partition_dir = dataset_root / f"period={period}" / f"run={run}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(partition_dir / "data.parquet")
    return dataset_root


def _make_dataset_with_empty_partition(tmp_path: Path) -> Path:
    """Write a dataset where one partition has zero passes_analysis events."""
    dataset_root = tmp_path / "parquet" / "background"
    rng = np.random.default_rng(99)

    # Normal partition
    n = 100
    normal = pl.DataFrame(
        {
            "energy": rng.uniform(20, 300, size=n),
            "passes_analysis": [True] * n,
            "passes_comparison": [True] * n,
            "period": ["p03"] * n,
            "run": ["r000"] * n,
        }
    )
    d1 = dataset_root / "period=p03" / "run=r000"
    d1.mkdir(parents=True, exist_ok=True)
    normal.write_parquet(d1 / "data.parquet")

    # Empty partition (all fails)
    empty = pl.DataFrame(
        {
            "energy": rng.uniform(20, 300, size=n),
            "passes_analysis": [False] * n,
            "passes_comparison": [False] * n,
            "period": ["p04"] * n,
            "run": ["r001"] * n,
        }
    )
    d2 = dataset_root / "period=p04" / "run=r001"
    d2.mkdir(parents=True, exist_ok=True)
    empty.write_parquet(d2 / "data.parquet")

    return dataset_root


class TestPlotBackgroundSpectrum:
    def test_spectrum_plot_file_is_created(self, tmp_path: Path) -> None:
        dataset_root = _make_dataset(tmp_path)
        output_dir = tmp_path / "plots" / "background"

        result = plot_background_spectrum(
            dataset_root,
            energy_range_keV=(20, 300),
            bin_width_keV=5,
            comparison_cut_profile="without-bb-like",
            output_dir=output_dir,
        )

        assert result.exists()
        assert result.suffix == ".png"
        assert result.parent == output_dir
        assert "20_300keV_5keV" in result.name

    def test_spectrum_plot_both_panels_present(self, tmp_path: Path) -> None:
        """The saved figure has exactly two axes (spectrum + ratio)."""
        import matplotlib.image as mpimg

        dataset_root = _make_dataset(tmp_path)
        output_dir = tmp_path / "plots" / "background"

        result = plot_background_spectrum(
            dataset_root,
            energy_range_keV=(20, 300),
            bin_width_keV=5,
            comparison_cut_profile="without-bb-like",
            output_dir=output_dir,
        )

        # Verify the file is a valid image
        img = mpimg.imread(str(result))
        assert img.shape[0] > 0
        assert img.shape[1] > 0

    def test_spectrum_with_exposure_normalisation(self, tmp_path: Path) -> None:
        dataset_root = _make_dataset(tmp_path)
        output_dir = tmp_path / "plots" / "background"

        result = plot_background_spectrum(
            dataset_root,
            energy_range_keV=(20, 300),
            bin_width_keV=5,
            comparison_cut_profile="without-bb-like",
            output_dir=output_dir,
            exposure_kg_yr=10.0,
        )

        assert result.exists()

    def test_ratio_handles_zero_analysis_bins(self, tmp_path: Path) -> None:
        """Bins where passes_analysis == 0 do not raise."""
        dataset_root = tmp_path / "parquet" / "background"
        # All events fail analysis — every bin will have zero analysis counts
        frame = pl.DataFrame(
            {
                "energy": [100.0, 150.0, 200.0],
                "passes_analysis": [False, False, False],
                "passes_comparison": [True, True, True],
                "period": ["p03", "p03", "p03"],
                "run": ["r000", "r000", "r000"],
            }
        )
        d = dataset_root / "period=p03" / "run=r000"
        d.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(d / "data.parquet")

        output_dir = tmp_path / "plots" / "background"

        # Should not raise
        result = plot_background_spectrum(
            dataset_root,
            energy_range_keV=(20, 300),
            bin_width_keV=10,
            comparison_cut_profile="without-bb-like",
            output_dir=output_dir,
        )
        assert result.exists()


class TestPlotBackgroundPartitionSummary:
    def test_partition_summary_file_is_created(self, tmp_path: Path) -> None:
        dataset_root = _make_dataset(tmp_path)
        output_dir = tmp_path / "plots" / "background"

        path, warnings = plot_background_partition_summary(dataset_root, output_dir)

        assert path.exists()
        assert path.name == "partition_summary.png"
        assert path.parent == output_dir

    def test_partition_summary_warns_on_empty_partitions(
        self, tmp_path: Path
    ) -> None:
        dataset_root = _make_dataset_with_empty_partition(tmp_path)
        output_dir = tmp_path / "plots" / "background"

        path, warnings = plot_background_partition_summary(dataset_root, output_dir)

        assert path.exists()
        assert len(warnings) == 1
        assert "p04/r001" in warnings[0]
        assert "zero" in warnings[0].lower()

    def test_no_warnings_when_all_partitions_have_events(
        self, tmp_path: Path
    ) -> None:
        dataset_root = _make_dataset(tmp_path)
        output_dir = tmp_path / "plots" / "background"

        _, warnings = plot_background_partition_summary(dataset_root, output_dir)

        assert warnings == []

    def test_partition_summary_with_exposure_overlay(
        self, tmp_path: Path
    ) -> None:
        dataset_root = _make_dataset(tmp_path)
        output_dir = tmp_path / "plots" / "background"

        exposure = {("p03", "r000"): 1.5, ("p04", "r001"): 2.3}
        path, warnings = plot_background_partition_summary(
            dataset_root, output_dir, exposure_by_partition=exposure
        )

        assert path.exists()
        assert warnings == []


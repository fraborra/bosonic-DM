# Copyright (C) 2026 Francesco Borra
#

"""``scripts/stan_grid_comparison.py`` on minimal pipeline outputs."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from signal_pipeline_outputs import OBSERVED, write_signal_pipeline_outputs

from bosonic_dm.yaml_io import read_yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stan_grid_comparison.py"
_spec = importlib.util.spec_from_file_location("stan_grid_comparison", SCRIPT)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

SAMPLER = ["--chains", "4", "--iter-warmup", "300", "--iter-sampling", "800"]


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    write_signal_pipeline_outputs(tmp_path)
    return tmp_path / "config.yaml"


def _run(config_path: Path, outdir: Path, signal_model: Any, *extra: str) -> None:
    script.main(
        [
            "--config",
            str(config_path),
            "--outdir",
            str(outdir),
            "--masses",
            "290,310",
            "--n-toys",
            "3",
            "--stan-exe",
            str(signal_model.exe_file),
            *SAMPLER,
            *extra,
        ]
    )


def test_main_writes_all_outputs(
    config_path: Path, tmp_path: Path, signal_model: Any
) -> None:
    outdir = tmp_path / "out"
    _run(config_path, outdir, signal_model)
    for name in (
        "stan_per_toy.parquet",
        "comparison.parquet",
        "comparison.png",
        "comparison.yaml",
        "per_mass/290keV_t0-2.parquet",
        "per_mass/310keV_t0-2.parquet",
    ):
        assert (outdir / name).is_file(), name
    record = read_yaml(outdir / "comparison.yaml")
    assert record["complete"] is True
    assert record["masses_present"] == [290.0, 310.0]
    assert record["masses_requested"] == [290.0, 310.0]
    assert record["masses_incomplete"] == []
    assert record["settings"]["n_toys"] == 3
    assert "passed_all" in record["agreement"]


def test_second_run_skips_the_fits(
    config_path: Path,
    tmp_path: Path,
    signal_model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outdir = tmp_path / "out"
    _run(config_path, outdir, signal_model)
    _no_fits(monkeypatch)
    _run(config_path, outdir, signal_model)


def _no_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_a: Any, **_k: Any) -> None:
        msg = "stan_limits must not be called"
        raise AssertionError(msg)

    monkeypatch.setattr(script, "stan_limits", fail)


def test_adding_toys_later_matches_one_run(
    config_path: Path, tmp_path: Path, signal_model: Any
) -> None:
    single = tmp_path / "single"
    _run(config_path, single, signal_model, "--n-toys", "5")
    added = tmp_path / "added"
    _run(config_path, added, signal_model, "--n-toys", "5", "--toys", "0:3")
    partial = read_yaml(added / "comparison.yaml")
    assert partial["complete"] is False
    assert partial["masses_incomplete"] == [290.0, 310.0]

    _run(config_path, added, signal_model, "--n-toys", "5", "--toys", "3:5")
    assert (added / "per_mass" / "290keV_t3-4.parquet").is_file()
    assert read_yaml(added / "comparison.yaml")["complete"] is True
    a = pl.read_parquet(single / "stan_per_toy.parquet")
    b = pl.read_parquet(added / "stan_per_toy.parquet")
    assert a.equals(b)

    # a smaller --n-toys compares only the first toys
    _run(config_path, added, signal_model, "--compare-only")
    assert pl.read_parquet(added / "stan_per_toy.parquet")["toy"].max() == 2


def test_old_one_file_per_mass_is_read(
    config_path: Path,
    tmp_path: Path,
    signal_model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outdir = tmp_path / "out"
    _run(config_path, outdir, signal_model)
    per_mass = outdir / "per_mass"
    for mass in ("290", "310"):
        (per_mass / f"{mass}keV_t0-2.parquet").rename(per_mass / f"{mass}keV.parquet")
    _no_fits(monkeypatch)
    _run(config_path, outdir, signal_model)
    assert read_yaml(outdir / "comparison.yaml")["complete"] is True


def test_overwrite_refits_only_the_requested_toys(
    config_path: Path, tmp_path: Path, signal_model: Any
) -> None:
    outdir = tmp_path / "out"
    _run(config_path, outdir, signal_model)
    before = pl.read_parquet(outdir / "stan_per_toy.parquet")
    _run(config_path, outdir, signal_model, "--overwrite", "--toys", "1:3")
    per_mass = outdir / "per_mass"
    assert pl.read_parquet(per_mass / "290keV_t0-2.parquet")["toy"].to_list() == [0]
    assert pl.read_parquet(per_mass / "290keV_t1-2.parquet")["toy"].to_list() == [1, 2]
    after = pl.read_parquet(outdir / "stan_per_toy.parquet")
    # same seeds: the refit gives the same limits
    assert before.equals(after)
    assert read_yaml(outdir / "comparison.yaml")["complete"] is True


def test_toys_saved_twice_are_rejected(
    config_path: Path, tmp_path: Path, signal_model: Any
) -> None:
    outdir = tmp_path / "out"
    _run(config_path, outdir, signal_model)
    per_mass = outdir / "per_mass"
    pl.read_parquet(per_mass / "290keV_t0-2.parquet").head(1).write_parquet(
        per_mass / "290keV_t0-0.parquet"
    )
    with pytest.raises(SystemExit, match="more than once"):
        _run(config_path, outdir, signal_model, "--compare-only")


def test_parse_toy_range() -> None:
    assert script.parse_toy_range("200:1000") == (200, 1000)
    for bad in ("5", "3:3", "-1:2", "a:b"):
        with pytest.raises(argparse.ArgumentTypeError):
            script.parse_toy_range(bad)


def test_toy_range_beyond_n_toys_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        script.parse_args(
            ["--config", "c.yaml", "--outdir", str(tmp_path), "--toys", "0:300"]
        )
    args = script.parse_args(["--config", "c.yaml", "--outdir", str(tmp_path)])
    assert args.toys == (0, 200)


def test_split_jobs_then_compare_only_match_one_job(
    config_path: Path, tmp_path: Path, signal_model: Any
) -> None:
    single = tmp_path / "single"
    split = tmp_path / "split"
    _run(config_path, single, signal_model)
    for index in ("0", "1"):
        _run(config_path, split, signal_model, "--n-jobs", "2", "--job-index", index)
    _run(config_path, split, signal_model, "--compare-only")
    a = pl.read_parquet(single / "comparison.parquet")
    b = pl.read_parquet(split / "comparison.parquet")
    assert a.equals(b)
    assert read_yaml(split / "comparison.yaml")["complete"] is True


def test_tampered_grid_table_is_rejected(
    config_path: Path, tmp_path: Path, signal_model: Any
) -> None:
    from bosonic_dm.config import load_analysis_config  # noqa: PLC0415
    from bosonic_dm.fitting.sensitivity import expected_limits  # noqa: PLC0415
    from bosonic_dm.pipeline.sensitivity import load_signal_inputs  # noqa: PLC0415

    inputs = load_signal_inputs(load_analysis_config(config_path))
    grid = expected_limits(
        [290.0, 310.0],
        inputs.backgrounds,
        n_toys=3,
        base_seed=20261117,
        n_bootstrap=0,
        **inputs.window_options,
    ).per_toy
    good = tmp_path / "good.parquet"
    grid.write_parquet(good)
    _run(config_path, tmp_path / "ok", signal_model, "--grid-per-toy", str(good))

    bad = tmp_path / "bad.parquet"
    grid.with_columns(pl.col("r90") * 1.01).write_parquet(bad)
    with pytest.raises(SystemExit, match="290"):
        _run(
            config_path, tmp_path / "bad_out", signal_model, "--grid-per-toy", str(bad)
        )


def test_stan_sees_the_toys_not_the_data(
    config_path: Path,
    tmp_path: Path,
    signal_model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[np.ndarray] = []
    real = script.stan_limits

    def spy(window: Any, counts: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        seen.append(np.array(counts))
        assert counts.shape[0] == 3
        assert not np.all(counts == OBSERVED)
        return real(window, counts, *args, **kwargs)

    monkeypatch.setattr(script, "stan_limits", spy)
    _run(config_path, tmp_path / "out", signal_model)

    from bosonic_dm.config import load_analysis_config  # noqa: PLC0415
    from bosonic_dm.fitting.sensitivity import (  # noqa: PLC0415
        draw_background_toys,
        stack_backgrounds,
    )
    from bosonic_dm.pipeline.sensitivity import load_signal_inputs  # noqa: PLC0415

    inputs = load_signal_inputs(load_analysis_config(config_path))
    _, b_hat = stack_backgrounds(inputs.backgrounds)
    toys = draw_background_toys(b_hat, 3, 20261117)
    assert len(seen) == 2
    for counts in seen:
        width = counts.shape[2]
        # the window is a contiguous slice of the full-spectrum toys
        starts = [
            i
            for i in range(toys.shape[2] - width + 1)
            if np.array_equal(toys[:, :, i : i + width], counts)
        ]
        assert starts


def test_split_masses_is_interleaved_disjoint_and_covering() -> None:
    masses = np.arange(10.0)
    parts = [script.split_masses(masses, 3, i) for i in range(3)]
    np.testing.assert_array_equal(parts[0], [0, 3, 6, 9])
    np.testing.assert_array_equal(np.sort(np.concatenate(parts)), masses)
    assert len(set(np.concatenate(parts))) == masses.size


@pytest.mark.parametrize("index", [-1, 3])
def test_split_masses_rejects_bad_index(index: int) -> None:
    with pytest.raises(ValueError, match="job"):
        script.split_masses(np.arange(5.0), 3, index)


def test_job_split_defaults_from_slurm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "SLURM_ARRAY_TASK_COUNT",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_ARRAY_TASK_MIN",
    ):
        monkeypatch.delenv(key, raising=False)
    assert script.parse_args(["--config", "c", "--outdir", "o"]).n_jobs == 1
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "4")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "12")
    monkeypatch.setenv("SLURM_ARRAY_TASK_MIN", "10")
    args = script.parse_args(["--config", "c", "--outdir", "o"])
    assert (args.n_jobs, args.job_index) == (4, 2)

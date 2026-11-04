# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

import numpy as np
import polars as pl

from bosonic_dm.background import prepare_background_spectra
from bosonic_dm.cuts import (
    build_usability_table,
    select_usable,
    usability_table_from_eres,
)


def _ch(usability: str, processable: bool = True, system: str = "geds") -> dict:
    return {
        "system": system,
        "analysis": {"usability": usability, "processable": processable},
    }


CHMAPS = {
    ("p03", "r000"): {
        "DET_ON": _ch("on"),
        "DET_OFF": _ch("off"),
        "DET_NOPROC": _ch("on", processable=False),
        "DET_SWITCH": _ch("on"),
        "SPM": _ch("on", system="spms"),
    },
    ("p03", "r001"): {
        "DET_ON": _ch("on"),
        "DET_OFF": _ch("off"),
        "DET_NOPROC": _ch("on", processable=False),
        "DET_SWITCH": _ch("ac"),
    },
}


def _events() -> pl.DataFrame:
    dets = ["DET_ON", "DET_OFF", "DET_NOPROC", "DET_SWITCH"]
    rows = [(p, r, d) for (p, r) in CHMAPS for d in dets]
    return pl.DataFrame(
        {
            "period": [r[0] for r in rows],
            "run": [r[1] for r in rows],
            "detector_name": [r[2] for r in rows],
            "energy": [100.0] * len(rows),
            "passes_analysis": [True] * len(rows),
            "passes_comparison": [True] * len(rows),
        }
    )


def test_build_usability_table_is_run_aware() -> None:
    calls = []

    def get(period: str, run: str) -> dict:
        calls.append((period, run))
        return CHMAPS[(period, run)]

    table = build_usability_table(
        get, [("p03", "r000"), ("p03", "r001"), ("p03", "r000")]
    )
    assert sorted(calls) == [("p03", "r000"), ("p03", "r001")]
    assert sorted(table.rows()) == [
        ("p03", "r000", "DET_ON"),
        ("p03", "r000", "DET_SWITCH"),
        ("p03", "r001", "DET_ON"),
    ]


def test_select_usable_keeps_only_usable_rows() -> None:
    usable = build_usability_table(lambda p, r: CHMAPS[(p, r)], CHMAPS)
    out = select_usable(_events(), usable).collect()
    assert out.columns == _events().columns
    assert sorted(out.select("run", "detector_name").rows()) == [
        ("r000", "DET_ON"),
        ("r000", "DET_SWITCH"),
        ("r001", "DET_ON"),
    ]


def test_select_usable_handles_categorical_partition_keys() -> None:
    events = _events().with_columns(pl.col("run").cast(pl.Categorical))
    usable = build_usability_table(lambda p, r: CHMAPS[(p, r)], CHMAPS)
    assert select_usable(events, usable).collect().height == 3


def test_usability_table_from_eres() -> None:
    eres = {"p03": {"r000": {"A": {"usability": "on"}, "B": {"usability": "ac"}}}}
    assert usability_table_from_eres(eres).rows() == [("p03", "r000", "A")]


def test_prepare_background_spectra_applies_usable() -> None:
    eres = {
        p: {
            r: {d: {"usability": "on", "expo": 1.0} for d in ("DET_ON", "DET_SWITCH")}
            for (pp, r) in CHMAPS
            if pp == p
        }
        for p in ("p03",)
    }
    groups = {"all": {"DET_ON": "all", "DET_SWITCH": "all"}}
    kwargs = {"energy_range_keV": (60.0, 200.0), "bin_width_keV": 10.0}
    usable = build_usability_table(lambda p, r: CHMAPS[(p, r)], CHMAPS)

    unfiltered = prepare_background_spectra(_events(), groups, eres, **kwargs)
    filtered = prepare_background_spectra(
        _events(), groups, eres, usable=usable, **kwargs
    )

    assert np.sum(unfiltered["all"].counts_analysis) == 4
    assert np.sum(filtered["all"].counts_analysis) == 3

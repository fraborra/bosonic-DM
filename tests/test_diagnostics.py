# Copyright (C) 2026 Francesco Borra
#

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from bosonic_dm.fitting.diagnostics import diagnostics_passed, sampler_diagnostics


class _FakeFit:
    """Minimal stand-in for ``cmdstanpy.CmdStanMCMC`` with well-mixed energies."""

    chains = 2
    num_draws_sampling = 200
    divergences = np.array([0, 0])
    max_treedepths = np.array([0, 0])

    def draws_pd(self, vars: list[str]) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        size = self.chains * self.num_draws_sampling
        return pd.DataFrame({vars[0]: rng.normal(size=size)})


def _parameters(**overrides: dict[str, float]) -> dict[str, dict[str, float]]:
    good = {"r_hat": 1.001, "ess_bulk": 900.0, "ess_tail": 800.0}
    parameters = {name: dict(good) for name in ("first", "middle", "last")}
    for name, values in overrides.items():
        parameters[name].update(values)
    return parameters


def test_good_fit_passes() -> None:
    diagnostics = sampler_diagnostics(_FakeFit(), _parameters())

    assert diagnostics["r_hat_max"] == pytest.approx(1.001)
    assert diagnostics["ess_bulk_min"] == pytest.approx(900.0)
    assert diagnostics_passed(diagnostics)


@pytest.mark.parametrize("position", ["first", "middle", "last"])
@pytest.mark.parametrize("key", ["r_hat", "ess_bulk", "ess_tail"])
def test_nan_fails_whatever_its_position(position: str, key: str) -> None:
    diagnostics = sampler_diagnostics(
        _FakeFit(), _parameters(**{position: {key: math.nan}})
    )

    assert not diagnostics_passed(diagnostics)


def test_infinite_r_hat_fails() -> None:
    diagnostics = sampler_diagnostics(_FakeFit(), _parameters(last={"r_hat": math.inf}))

    assert not diagnostics_passed(diagnostics)


def test_thresholds_still_apply() -> None:
    low_ess = sampler_diagnostics(_FakeFit(), _parameters(middle={"ess_tail": 100.0}))
    high_r_hat = sampler_diagnostics(_FakeFit(), _parameters(first={"r_hat": 1.05}))

    assert not diagnostics_passed(low_ess)
    assert not diagnostics_passed(high_r_hat)
    assert diagnostics_passed(low_ess, ess_min=50.0)

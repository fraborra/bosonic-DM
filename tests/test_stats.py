# Copyright (C) 2025 Francesco Borra
#

from __future__ import annotations

import pytest

from bosonic_dm.stats import bayesian_efficiency


def test_bayesian_efficiency_accepts_zero_successes() -> None:
    mean, sigma = bayesian_efficiency(0, 100)

    assert 0 < mean < 0.01
    assert sigma > 0


@pytest.mark.parametrize(
    ("successes", "trials"),
    [
        (0, 0),
        (-1, 10),
        (11, 10),
    ],
)
def test_bayesian_efficiency_rejects_invalid_counts(
    successes: int,
    trials: int,
) -> None:
    with pytest.raises(ValueError):
        bayesian_efficiency(successes, trials)

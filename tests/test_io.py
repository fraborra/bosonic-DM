# Copyright (C) 2025 Francesco Borra
#

from __future__ import annotations

import numpy as np
import pytest

from bosonic_dm.io import get_mean_fcc_det_group


def test_get_mean_fcc_det_group_accepts_yaml_shaped_groups() -> None:
    ratio_dict = {
        200: {
            "V00001A": {"ratio": 1.0},
            "V00002A": {"ratio": 3.0},
            "B00001A": {"ratio": 5.0},
        }
    }
    detector_groups = {
        "ICPC group1": {
            "V00001A": {"~p09": "all"},
            "V00002A": {"p09": "all"},
        },
        "mixed": ["V00002A", "B00001A"],
    }
    eres_dict = {
        "p08": {
            "r001": {
                "V00001A": {"usability": "on", "expo": 1.0},
                "V00002A": {"usability": "on", "expo": 2.0},
                "B00001A": {"usability": "on", "expo": 4.0},
            }
        },
        "p09": {
            "r001": {
                "V00001A": {"usability": "on", "expo": 10.0},
                "V00002A": {"usability": "on", "expo": 20.0},
                "B00001A": {"usability": "on", "expo": 40.0},
            },
            "r002": {
                "V00002A": {"usability": "off", "expo": 200.0},
            },
        },
    }

    result = get_mean_fcc_det_group(ratio_dict, detector_groups, eres_dict)

    assert result[200]["ICPC group1"] == pytest.approx(61.0 / 21.0)
    assert result[200]["mixed"] == pytest.approx(286.0 / 66.0)


def test_get_mean_fcc_det_group_matches_filtering_and_uncertainty_logic() -> None:
    ratio_dict = {
        200: {
            "V00001A": {"value": 1.0, "sigma": 0.1},
            "V00002A": {"value": 3.0, "sigma": 0.2},
            "V00003A": {"value": np.nan, "sigma": 0.3},
            "V00004A": {"value": 10.0, "sigma": 0.4},
        }
    }
    detector_groups = {
        "selected": ["V00001A", "V00002A", "V00003A", "V00004A"],
        "overlap": ["V00002A"],
        "empty": ["V99999A"],
    }
    eres_dict = {
        "p03": {
            "r000": {
                "V00001A": {"usability": "on", "weight": 1.0},
                "V00002A": {"usability": "on", "weight": 3.0},
                "V00003A": {"usability": "on", "weight": 4.0},
                "V00004A": {"usability": "on", "weight": 2.0},
            }
        }
    }

    result = get_mean_fcc_det_group(
        ratio_dict,
        detector_groups,
        eres_dict,
        key="value",
        weight_key="weight",
        unc_key="sigma",
        exclude_dets=["V00004A"],
    )

    expected_mean = 2.5
    expected_measurement_unc = np.sqrt(0.1**2 + (3.0 * 0.2) ** 2) / 4.0
    expected_scatter_unc = np.sqrt(0.75 / 1.6)
    expected_unc = np.sqrt(expected_measurement_unc**2 + expected_scatter_unc**2)

    assert result[200]["selected"] == {
        "value": pytest.approx(expected_mean),
        "unc": pytest.approx(expected_unc),
        "exposure": pytest.approx(4.0),
    }
    assert result[200]["overlap"] == {
        "value": pytest.approx(3.0),
        "unc": pytest.approx(0.2),
        "exposure": pytest.approx(3.0),
    }
    assert "empty" not in result[200]


def test_get_mean_fcc_det_group_rejects_string_membership() -> None:
    with pytest.raises(TypeError, match="must be a sequence or mapping"):
        get_mean_fcc_det_group({}, {"invalid": "V00001A"}, {})

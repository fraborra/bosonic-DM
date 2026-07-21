# Copyright (C) 2025 Francesco Borra
#

from __future__ import annotations

import matplotlib.pyplot as plt

from bosonic_dm.plotting.efficiency import plot_efficiency_comparison


def test_plot_efficiency_comparison_supports_detector_groups() -> None:
    groups = ["ICPC group1", "ICPC group2", "BEGe", "PPC", "COAX"]
    means = {
        200: {group: {"value": 0.8, "unc": 0.01, "exposure": 1.0} for group in groups}
    }
    labels = {"all": ("All", "-", "o", means)}

    fig, axes = plot_efficiency_comparison(
        labels,
        interaction="axio-electric",
        plot_type="efficiency",
        plot_title="Efficiency comparison",
        ylabel="Efficiency",
        group_by="detector_group",
    )

    visible_titles = [ax.get_title() for ax in axes.flat if ax.get_visible()]
    assert visible_titles == groups
    assert axes.shape == (3, 2)
    assert not axes.flat[-1].get_visible()
    plt.close(fig)

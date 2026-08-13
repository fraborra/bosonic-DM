# Copyright (C) 2025 Francesco Borra
#

"""Small shared helpers for bosonic-DM."""

from __future__ import annotations

import argparse

import numpy as np

DEFAULT_MASS_POINTS_KEV: tuple[int, ...] = tuple(range(200, 1001, 100))


def add_mass_selection_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard ``-e``/``--mass-range`` mutually-exclusive group.

    Every CLI that iterates over DM mass points should call this once,
    then pass the parsed ``args`` to :func:`resolve_mass_points`.
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-e",
        "--energies",
        nargs="+",
        type=int,
        help="Explicit list of m_DM values in keV (e.g. -e 200 300 400).",
    )
    group.add_argument(
        "--mass-range",
        nargs=3,
        type=int,
        metavar=("START_KEV", "STOP_KEV", "STEP_KEV"),
        help="Inclusive mass range in keV (e.g. --mass-range 100 1020 10).",
    )


def resolve_mass_points(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[int, ...]:
    """Turn parsed mass-selection arguments into a concrete tuple of mass points.

    Parameters
    ----------
    args
        Namespace returned by ``parser.parse_args()``.
    parser
        The parser instance, used to call ``parser.error()`` on bad input.

    Returns
    -------
    tuple of int
        Mass points in keV.
    """
    if args.mass_range is not None:
        start, stop, step = args.mass_range
        if start <= 0 or stop <= 0 or step <= 0:
            parser.error("mass-range values must all be positive")
        if stop < start:
            parser.error("STOP_KEV must be >= START_KEV")
        if (stop - start) % step != 0:
            parser.error("STOP_KEV must lie on the requested mass-range step")
        return tuple(range(start, stop + 1, step))
    if args.energies is not None:
        return tuple(args.energies)
    return DEFAULT_MASS_POINTS_KEV


def calculate_energies(m_dm: float) -> tuple[float, float]:
    """Calculate Dark-Compton electron recoil and photon energies.

    Non-relativistic approximation where the DM kinetic energy is
    negligible compared to its rest mass (omega ≈ m_DM).

    Parameters
    ----------
    m_dm
        Dark matter particle mass in keV.

    Returns
    -------
    tuple of float
        ``(T, omega_prime)`` — electron recoil kinetic energy and
        outgoing photon energy, both in keV.
    """
    m_e = 510.99895  # electron mass [keV]
    T = m_dm**2 / (2 * (m_e + m_dm))
    omega_prime = np.sqrt(T**2 + 2 * m_e * T)
    return T, omega_prime


def get_energy_interval(m_dm: float, region: str) -> tuple[float, float]:
    """Calculate the recoil energy interval for a given dark matter mass.

    Parameters
    ----------
    m_dm
        Dark matter mass in keV.
    region
        Selected region interval (``"low"`` or ``"mid"``).

    Returns
    -------
    tuple of float
        The calculated (low, high) energy interval bounds in keV.
    """
    T, omega_prime = calculate_energies(m_dm)
    if region == "low":
        return 25.0, T - 10.0
    if region == "mid":
        return T + 10.0, omega_prime - 10.0
    msg = f"Unknown region '{region}'; expected 'low' or 'mid'"
    raise ValueError(msg)


def expand_range(item):
    if ".." not in item:
        return [item]  # already single
    start, end = item.split("..")

    prefix = start[0]  # 'r'
    s = int(start[1:])
    e = int(end[1:])
    width = len(start) - 1  # number of digits

    return [f"{prefix}{i:0{width}d}" for i in range(s, e + 1)]


def clean_array(arr):
    arr = np.asarray(arr)
    arr = arr.astype(float)
    return arr[(arr != 0) & np.isfinite(arr)]


def select_channel(energies, channels, rawid):
    return energies[channels == rawid]

# Copyright (C) 2025 Francesco Borra
#

"""Dark Compton kinematics and GPS macro generation.

Usage
-----
After installing the package (``pixi run pip install -e .``), this module
is available as a CLI tool::

    pixi run generate-dark-compton -e 200 300 400

This generates the GPS macro YAML config files in the ``tmp/`` directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def calculate_energies(m_dm: float) -> tuple[float, float]:
    """
    Calcola l'energia di rinculo dell'elettrone e il fotone emesso.

    Calcola l'energia di rinculo dell'elettrone (T) e l'energia del fotone emesso (omega')
    dato uno scenario non relativistico in cui l'energia della particella X è circa la sua massa (omega ~ m_DM).

    Equazioni di riferimento dalla tesi di Sofia:
    (1.35): T = m_DM^2 / (2 * (m_e + m_DM))
    (1.36): omega' = sqrt(T^2 + 2 * m_e * T)

    Args:
        m_dm (float or numpy.ndarray): Massa della particella chi (X) in keV.

    Returns
    -------
        tuple: (T, omega_prime)
            - T (float or numpy.ndarray): Energia di rinculo dell'elettrone in keV.
            - omega_prime (float or numpy.ndarray): Energia del fotone uscente in keV.
    """
    m_e = 510.99895  # mass of the electron in keV

    # Eq. (1.35) - Energia di rinculo dell'elettrone
    T = (m_dm**2) / (2 * (m_e + m_dm))

    # Eq. (1.36) - Energia del fotone uscente
    omega_prime = np.sqrt(T**2 + 2 * m_e * T)

    return T, omega_prime


def get_energy_interval(m_dm: float, region: str) -> tuple[float, float]:
    """Calculate the recoil energy interval for a given dark matter mass.

    Parameters
    ----------
    m_dm : float
        Dark matter mass (sim_e) in keV.
    region : {"low", "mid"}
        Selected region interval ("low" or "mid").

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dark compton GPS macros.")
    parser.add_argument(
        "-e",
        "--energies",
        nargs="+",
        type=int,
        help="List of m_dm values in keV. If not provided, uses range(200, 1100, 100).",
    )
    args = parser.parse_args()

    simulated_energies = args.energies or range(200, 1100, 100)

    output_dir = "tmp"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    generators_file = f"{output_dir}/generators-dark_compton.yaml"
    simconfig_file = f"{output_dir}/simconfig-dark_compton.yaml"
    with (
        Path.open(generators_file, "w") as f_gen,
        Path.open(simconfig_file, "w") as f_sim,
    ):
        for m_dm in simulated_energies:
            T, omega_prime = calculate_energies(m_dm)

            # Write generator config
            f_gen.write(f"dark_compton_{m_dm}keV:\n")
            f_gen.write("  - /RMG/Generator/Select GPS\n")
            f_gen.write("  - /gps/particle e-\n")
            f_gen.write(f"  - /gps/energy {T:.4f} keV\n")
            f_gen.write("  - /gps/ang/type iso\n")
            f_gen.write("  - /gps/source/add 1\n")
            f_gen.write("  - /gps/particle gamma\n")
            f_gen.write(f"  - /gps/energy {omega_prime:.4f} keV\n")
            f_gen.write("  - /gps/ang/type iso\n")
            f_gen.write("  - /gps/source/multiplevertex true\n\n")

            # Write simconfig
            f_sim.write(f"dark_compton_{m_dm}keV_hpge_bulk:\n")
            f_sim.write("  template: $_/templates/default.mac\n")
            f_sim.write(f"  generator: ~defines:dark_compton_{m_dm}keV\n")
            f_sim.write("  confinement: ~volumes.bulk:^[VBCP].*\n")
            f_sim.write("  primaries_per_job: 2000000\n")
            f_sim.write("  number_of_jobs: 5\n\n")


if __name__ == "__main__":
    main()

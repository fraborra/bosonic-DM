# Copyright (C) 2025 Francesco Borra
#

"""GPS macro generation for remage simulation configurations.

Usage
-----
After installing the package (``pixi run pip install -e .``), this module
is available as a CLI tool::

    pixi run generate-gps-macros -e 200 300 400

This generates the GPS macro YAML config files in the ``tmp/`` directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bosonic_dm.utils import add_mass_selection_args, resolve_mass_points


def main() -> None:
    """CLI entry point for GPS macro generation."""
    parser = argparse.ArgumentParser(description="Generate GPS macros for remage.")
    add_mass_selection_args(parser)
    parser.add_argument(
        "-t",
        "--type",
        choices=["dark_compton", "axio_electric", "both"],
        default="both",
        help="Type of macros to generate.",
    )
    args = parser.parse_args()

    simulated_energies = resolve_mass_points(args, parser)

    output_dir = "tmp"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if args.type in ("dark_compton", "both"):
        generators_dc_file = f"{output_dir}/generators-dark_compton.yaml"
        simconfig_dc_file = f"{output_dir}/simconfig-dark_compton.yaml"
        with (
            Path.open(generators_dc_file, "w") as f_gen_dc,
            Path.open(simconfig_dc_file, "w") as f_sim_dc,
        ):
            for m_dm in simulated_energies:
                # Write dark compton generator config
                f_gen_dc.write(f"fromfile_dark_compton_{m_dm}keV:\n")
                f_gen_dc.write("  - /RMG/Generator/Select FromFile\n")
                f_gen_dc.write(
                    f"  - /RMG/Generator/FromFile/FileName /pscratch/sd/b/borrfran/sim-v1.1.0-20260401/inputs/simprod/config/tier/stp/l200cfg01/kinematics/dark_compton_{m_dm}keV.lh5\n\n"
                )

                # Write dark compton simconfig
                f_sim_dc.write(f"fromfile_dark_compton_{m_dm}keV_hpge_bulk:\n")
                f_sim_dc.write("  template: $_/templates/default.mac\n")
                f_sim_dc.write(
                    f"  generator: ~defines:fromfile_dark_compton_{m_dm}keV\n"
                )
                f_sim_dc.write("  confinement: ~volumes.bulk:^[VBCP].*\n")
                f_sim_dc.write("  primaries_per_job: 2000000\n")
                f_sim_dc.write("  number_of_jobs: 20\n\n")

    if args.type in ("axio_electric", "both"):
        generators_axio_file = f"{output_dir}/generators-axio_electric.yaml"
        simconfig_axio_file = f"{output_dir}/simconfig-axio_electric.yaml"
        with (
            Path.open(generators_axio_file, "w") as f_gen_axio,
            Path.open(simconfig_axio_file, "w") as f_sim_axio,
        ):
            for m_dm in simulated_energies:
                # Write axio-electric generator config
                f_gen_axio.write(f"electrons_{m_dm}keV:\n")
                f_gen_axio.write("  - /RMG/Generator/Select GPS\n")
                f_gen_axio.write("  - /gps/particle e-\n")
                f_gen_axio.write(f"  - /gps/energy {m_dm} keV\n")
                f_gen_axio.write("  - /gps/ang/type iso\n\n")

                # Write axio-electric simconfig
                f_sim_axio.write(f"electron_{m_dm}keV_hpge_bulk:\n")
                f_sim_axio.write("  template: $_/templates/default.mac\n")
                f_sim_axio.write(f"  generator: ~defines:electrons_{m_dm}keV\n")
                f_sim_axio.write("  confinement: ~volumes.bulk:^[VBCP].*\n")
                f_sim_axio.write("  primaries_per_job: 2000000\n")
                f_sim_axio.write("  number_of_jobs: 20\n\n")


if __name__ == "__main__":
    main()

# Copyright (C) 2025 Francesco Borra
#

"""Generate correlated Dark Compton kinematics for remage FromFile generator.

Produces an LH5 file with the kinematic table (vtx/kin) where each event
contains exactly 2 primary particles (e- + gamma) sharing the same vertex.
The vertex position is NOT embedded in this file — it is handled by remage's
built-in confinement system, which applies the confined position to all
particles in the event.

Format (as required by RMGGeneratorFromFile)::

    vtx/kin:
      g4_pid  (int64)   - PDG code: 11 for e-, 22 for gamma
      ekin    (float64) - kinetic energy [keV], stored with units attr
      px, py, pz (float64) - unit momentum direction (isotropic)
      time    (float64) - always 0 [ns]
      n_part  (int64)   - 2 for first row of each event, 0 for second row

Usage
-----
After installing the package (``pixi run pip install -e .``), this module
is available as a CLI tool::

    pixi run generate-dark-compton-kin -e 200 --events 10000000
    pixi run generate-dark-compton-kin --mass-range 100 1020 10
    pixi run generate-dark-compton-kin  # generates all default mass points

The electron and photon energies are computed from the Dark-Compton kinematics
in :func:`bosonic_dm.utils.calculate_energies`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from lgdo import Array, Table, lh5

from bosonic_dm.utils import (
    add_mass_selection_args,
    calculate_energies,
    resolve_mass_points,
)

logger = logging.getLogger(__name__)

# PDG codes
PDG_ELECTRON = 11
PDG_GAMMA = 22


def random_isotropic_direction(
    n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample *n* isotropic unit vectors.

    Uses the standard rejection-free method: cos(theta) uniform in [-1, 1],
    phi uniform in [0, 2*pi].

    Parameters
    ----------
    n
        Number of vectors to sample.
    rng
        NumPy random generator instance.

    Returns
    -------
    tuple of ndarray
        ``(px, py, pz)`` arrays of shape ``(n,)``.
    """
    cos_theta = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    px = sin_theta * np.cos(phi)
    py = sin_theta * np.sin(phi)
    pz = cos_theta
    return px, py, pz


def generate_dark_compton_kin(
    output_path: str | Path,
    n_events: int,
    e_electron_kev: float,
    e_gamma_kev: float,
    seed: int = 0,
    chunk_size: int = 500_000,
) -> None:
    """Write an LH5 kinematic file for Dark Compton (e- + gamma) events.

    Parameters
    ----------
    output_path
        Output .lh5 file path.
    n_events
        Number of events (each event = 1 e- + 1 gamma).
    e_electron_kev
        Kinetic energy of the electron [keV].
    e_gamma_kev
        Kinetic energy of the photon [keV].
    seed
        Random seed for reproducibility.
    chunk_size
        Number of events to hold in memory at once.  Chunks are appended
        internally using one continuously advancing random-number generator.
    """
    if n_events <= 0:
        msg = "n_events must be positive"
        raise ValueError(msg)
    if chunk_size <= 0:
        msg = "chunk_size must be positive"
        raise ValueError(msg)

    output_path = Path(output_path)
    rng = np.random.default_rng(seed)

    for start in range(0, n_events, chunk_size):
        n_events_chunk = min(chunk_size, n_events - start)
        n_rows = 2 * n_events_chunk

        # Alternating: e-, gamma, e-, gamma, ...
        g4_pid = np.empty(n_rows, dtype=np.int64)
        g4_pid[0::2] = PDG_ELECTRON
        g4_pid[1::2] = PDG_GAMMA

        ekin = np.empty(n_rows, dtype=np.float64)
        ekin[0::2] = e_electron_kev
        ekin[1::2] = e_gamma_kev

        # Sample one isotropic event axis and assign opposite directions to the
        # electron and photon. The generator is deliberately not re-seeded per
        # chunk, so every chunk receives new, statistically independent axes.
        px_e, py_e, pz_e = random_isotropic_direction(n_events_chunk, rng)
        px = np.empty(n_rows, dtype=np.float64)
        py = np.empty(n_rows, dtype=np.float64)
        pz = np.empty(n_rows, dtype=np.float64)
        px[0::2], px[1::2] = px_e, -px_e
        py[0::2], py[1::2] = py_e, -py_e
        pz[0::2], pz[1::2] = pz_e, -pz_e

        table = Table(
            {
                "g4_pid": Array(g4_pid),
                "ekin": Array(ekin, attrs={"units": "keV"}),
                "px": Array(px),
                "py": Array(py),
                "pz": Array(pz),
                "time": Array(
                    np.zeros(n_rows, dtype=np.float64), attrs={"units": "ns"}
                ),
                "n_part": Array(
                    np.tile(np.array([2, 0], dtype=np.int64), n_events_chunk)
                ),
            }
        )
        wo_mode = "overwrite_file" if start == 0 else "append"
        lh5.write(
            table,
            name="kin",
            group="vtx",
            lh5_file=str(output_path),
            wo_mode=wo_mode,
        )

    logger.info(
        "Written %d events (%d rows) to %s\n"
        "  e-    Ekin = %.5f keV\n"
        "  gamma Ekin = %.5f keV\n"
        "  chunk size = %d events",
        n_events,
        2 * n_events,
        output_path,
        e_electron_kev,
        e_gamma_kev,
        chunk_size,
    )


def main() -> None:
    """CLI entry point for Dark Compton kinematic file generation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Generate Dark Compton kinematic input for remage FromFile generator."
    )
    add_mass_selection_args(parser)
    parser.add_argument(
        "--events",
        type=int,
        default=10_000_000,
        help="Number of events per file (default: 10_000_000).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("tmp"),
        help="Output directory for .lh5 files (default: tmp/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed (default: 0). Each mass point gets seed+i.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Events written per internal LH5 chunk (default: 500000).",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    mass_points_kev = resolve_mass_points(args, parser)

    for i, mass_kev in enumerate(mass_points_kev):
        e_e, e_g = calculate_energies(float(mass_kev))
        outfile = args.outdir / f"dark_compton_{mass_kev}keV.lh5"
        generate_dark_compton_kin(
            output_path=outfile,
            n_events=args.events,
            e_electron_kev=e_e,
            e_gamma_kev=e_g,
            seed=args.seed + i,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()

# Copyright (C) 2026 Francesco Borra
#

"""Print the memory used while counting primaries, one STP job at a time.

Runs the same steps as the primary counts of the efficiencies stage and
prints, after each job, the number of vertices, the current RSS and the peak
RSS. A current RSS that grows from job to job means memory is not released::

    pixi run python3 scripts/profile_primary_counts.py \
        --sim-root "$BOSONIC_DM_SIM_ROOT" --energies 100 110 --max-jobs 5
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import lh5

from bosonic_dm.geometry import (
    aggregate_vertex_counts_by_run,
    assign_detectors_to_vertices,
    build_detector_map,
    read_run_ranges,
)


def _current_rss_gb() -> float:
    """Return the resident memory of this process now, in GB (Linux only)."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024**2
    return float("nan")


def _peak_rss_gb() -> float:
    """Return the peak resident memory of this process, in GB (Linux: KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _report(label: str) -> None:
    print(  # noqa: T201 - a command-line report
        f"{label:<40} rss = {_current_rss_gb():6.2f} GB  "
        f"peak = {_peak_rss_gb():6.2f} GB",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--job-template", default="electron_{energy}keV_hpge_bulk")
    parser.add_argument("--energies", type=int, nargs="+", default=[100])
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args()

    tier_root = args.sim_root / "generated" / "tier"
    first_job = args.job_template.format(energy=args.energies[0])
    gdml = (
        args.sim_root
        / "generated/pars/geom"
        / f"l200cfg01-{first_job}-tier_stp-geom.gdml"
    )

    _report("start")
    det_map = build_detector_map(gdml)
    _report(f"geometry loaded ({len(det_map)} detectors)")

    for energy in args.energies:
        job_name = args.job_template.format(energy=energy)
        stp_files = sorted((tier_root / "stp" / job_name).glob("*-tier_stp.lh5"))
        stp_files = stp_files[: args.max_jobs]
        for stp_file in stp_files:
            evt_file = (
                tier_root
                / "evt"
                / job_name
                / stp_file.name.replace("tier_stp", "tier_evt")
            )
            n_vertices = lh5.read_n_rows("vtx", str(stp_file))
            start = time.monotonic()
            names, evtids = assign_detectors_to_vertices(
                gdml=gdml,
                lh5_files=stp_file,
                save=False,
                return_evtids=True,
                det_map=det_map,
            )
            _report(f"{energy} keV {stp_file.stem[-17:-9]} assigned")
            aggregate_vertex_counts_by_run(
                [names], [evtids], [read_run_ranges(evt_file)]
            )
            del names, evtids
            _report(
                f"{energy} keV {stp_file.stem[-17:-9]} done "
                f"({n_vertices / 1e6:.1f}M vtx, {time.monotonic() - start:.0f} s)"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

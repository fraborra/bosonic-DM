# Copyright (C) 2025 Francesco Borra
#

"""HPGe detector geometry utilities for vertex-to-detector assignment."""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import lgdo
import lh5
import numpy as np
import pyg4ometry as pg4
import pygeomhpges as hpges
from dbetto import Props
from numpy.typing import NDArray
from pygeomtools import get_sensvol_metadata

logger = logging.getLogger(__name__)

# Physical-volume name prefixes identifying HPGe detectors in the L200 GDML.
_HPGE_PREFIXES = ("V", "P", "B", "C")

# Conversion factor from metres (LH5 vertex coordinates) to mm (pygeomhpges).
_M_TO_MM = 1000.0


def build_detector_map(gdml: str | Path) -> dict:
    """Load an L200 GDML geometry and build HPGe detector objects.

    For each HPGe physical volume found in the GDML, this function builds the
    ``pygeomhpges`` solid, records its global position, and pre-computes a
    bounding cylinder for fast spatial pre-filtering.

    Parameters
    ----------
    gdml
        Path to the GDML file describing the L200 geometry.

    Returns
    -------
    dict
        Dictionary keyed by detector name (e.g. ``"V02160A"``), each value
        being a dict with keys:

        - ``"pos"`` — global position as a length-3 array (mm).
        - ``"hpge"`` — the ``pygeomhpges`` HPGe logical-volume object.
        - ``"r_max"`` — bounding-cylinder radius (mm).
        - ``"z_min"`` — bounding-cylinder lower z bound (mm, local frame).
        - ``"z_max"`` — bounding-cylinder upper z bound (mm, local frame).
    """
    gdml = str(gdml)
    reg = pg4.gdml.Reader(gdml).getRegistry()
    reg_tmp = pg4.geant4.Registry()

    detectors = [
        name for name in reg.physicalVolumeDict if name and name[0] in _HPGE_PREFIXES
    ]
    logger.debug("Found %d HPGe detectors in GDML", len(detectors))

    det_map: dict = {}
    for det_name in detectors:
        metadata = get_sensvol_metadata(reg, det_name)
        if metadata is None:
            logger.warning("No metadata for %s, skipping", det_name)
            continue

        hpge = hpges.make_hpge(
            metadata,
            name=det_name,
            registry=reg_tmp,
            allow_cylindrical_asymmetry=False,
        )
        pos = np.asarray(reg.physicalVolumeDict[det_name].position.eval())

        # Compute bounding cylinder from the polycone profile (r, z) in mm.
        r_profile, z_profile = hpge.get_profile()
        r_arr = np.asarray(r_profile, dtype=float)
        z_arr = np.asarray(z_profile, dtype=float)

        det_map[det_name] = {
            "pos": pos,
            "hpge": hpge,
            "r_max": float(np.max(r_arr)),
            "z_min": float(np.min(z_arr)),
            "z_max": float(np.max(z_arr)),
        }

    return det_map


def _process_single_file(
    lh5_file: str | Path,
    det_map: Mapping,
    vtx_group: str = "vtx",
    *,
    save: bool = False,
    return_evtids: bool = False,
) -> NDArray | tuple[NDArray, NDArray]:
    """Assign detectors for a single LH5 file using a pre-built detector map.

    Parameters
    ----------
    lh5_file
        Path to the LH5 file containing vertex data.
    det_map
        Detector map as returned by :func:`build_detector_map`.
    vtx_group
        HDF5 group name for the vertex data.
    save
        Whether to write the detector names back to the LH5 file.

    Returns
    -------
    NDArray
        String array with a detector name per vertex (``"none"`` if unmatched).
    """
    # Read vertex coordinates (stored in metres).
    xloc = np.asarray(lh5.read_as(f"{vtx_group}/xloc", lh5_file, "np"))
    yloc = np.asarray(lh5.read_as(f"{vtx_group}/yloc", lh5_file, "np"))
    zloc = np.asarray(lh5.read_as(f"{vtx_group}/zloc", lh5_file, "np"))

    if return_evtids:
        try:
            evtids = np.asarray(lh5.read_as(f"{vtx_group}/evtid", lh5_file, "np"))
        except Exception as e:
            logger.warning(
                "Failed to read evtid from %s, falling back to index: %s", lh5_file, e
            )
            evtids = np.arange(len(xloc))

    n_vertices = len(xloc)
    logger.debug("Read %d vertices from %s", n_vertices, lh5_file)

    # Convert to mm for pygeomhpges.
    x_mm = xloc * _M_TO_MM
    y_mm = yloc * _M_TO_MM
    z_mm = zloc * _M_TO_MM

    # Result array: "none" for unmatched vertices.
    det_names = np.full(n_vertices, "none", dtype="S8")
    # Track which vertices are still unassigned.
    unassigned = np.ones(n_vertices, dtype=bool)

    for det_name, info in det_map.items():
        if not np.any(unassigned):
            logger.debug("All vertices assigned, stopping early")
            break

        pos = info["pos"]
        hpge_obj = info["hpge"]
        r_max = info["r_max"]
        z_min = info["z_min"]
        z_max = info["z_max"]

        # Compute local coordinates for unassigned vertices only.
        idx_unassigned = np.flatnonzero(unassigned)
        lx = x_mm[idx_unassigned] - pos[0]
        ly = y_mm[idx_unassigned] - pos[1]
        lz = z_mm[idx_unassigned] - pos[2]

        # Bounding-cylinder pre-filter (vectorised, very cheap).
        r_sq = lx * lx + ly * ly
        in_cylinder = (r_sq <= r_max * r_max) & (lz >= z_min) & (lz <= z_max)

        n_candidates = int(np.sum(in_cylinder))
        if n_candidates == 0:
            continue

        # Build (N_cand, 3) array for candidates only.
        idx_candidates = idx_unassigned[in_cylinder]
        local_coords = np.column_stack(
            (lx[in_cylinder], ly[in_cylinder], lz[in_cylinder])
        )

        inside = hpge_obj.is_inside(local_coords)
        n_inside = int(np.sum(inside))

        if n_inside > 0:
            matched_idx = idx_candidates[inside]
            det_names[matched_idx] = det_name
            unassigned[matched_idx] = False
            logger.debug(
                "%s: %d candidates, %d inside", det_name, n_candidates, n_inside
            )

    n_assigned = int(np.sum(~unassigned))
    logger.debug(
        "Assigned %d / %d vertices (%.1f%%)",
        n_assigned,
        n_vertices,
        100.0 * n_assigned / max(n_vertices, 1),
    )

    # Write the detector names back to the LH5 file.
    if save:
        det_lgdo = lgdo.Array(nda=det_names)
        lh5.write(det_lgdo, "det", str(lh5_file), group=vtx_group, wo_mode="append")
        logger.debug("Wrote '%s/det' to %s", vtx_group, lh5_file)

    if return_evtids:
        return det_names, evtids
    return det_names


def aggregate_vertex_counts(
    det_name_arrays: Sequence[NDArray],
    evtid_arrays: Sequence[NDArray] | None = None,
) -> dict[str, int]:
    """Aggregate per-detector vertex counts across multiple files.

    Parameters
    ----------
    det_name_arrays
        Sequence of string arrays as returned by
        :func:`_process_single_file` or :func:`assign_detectors_to_vertices`.
        Each element is an array of detector name bytes (e.g. ``b"V02160A"``).
        Vertices labelled ``b"none"`` are excluded.
    evtid_arrays
        Optional sequence of event ID arrays (same length as ``det_name_arrays``).
        If provided, multiple vertices in the same detector from the same event
        will be counted only once.

    Returns
    -------
    dict[str, int]
        Dictionary mapping detector name (str) to the total number of vertices
        generated inside that detector, sorted alphabetically by detector name.
    """
    total: Counter[str] = Counter()

    if evtid_arrays is None:
        evtid_arrays = [None] * len(det_name_arrays)

    for arr, evtid_arr in zip(det_name_arrays, evtid_arrays, strict=True):
        if evtid_arr is not None:
            if len(arr) > 0:
                struct_arr = np.rec.fromarrays([evtid_arr, arr])
                unique_structs = np.unique(struct_arr)
                names, counts = np.unique(unique_structs.f1, return_counts=True)
            else:
                names, counts = [], []
        else:
            names, counts = np.unique(arr, return_counts=True)

        for name_bytes, count in zip(names, counts, strict=True):
            name_str = (
                name_bytes.decode()
                if isinstance(name_bytes, bytes)
                else str(name_bytes)
            )
            if name_str != "none":
                total[name_str] += int(count)

    return dict(sorted(total.items()))


def assign_detectors_to_vertices(
    gdml: str | Path,
    lh5_files: str | Path | Sequence[str | Path],
    vtx_group: str = "vtx",
    *,
    save: bool = False,
    counts_yaml: str | Path | None = None,
    return_evtids: bool = False,
) -> NDArray | list[NDArray] | tuple[NDArray | list[NDArray], NDArray | list[NDArray]]:
    """Determine which HPGe detector each vertex is inside.

    The geometry is loaded once from *gdml* and reused across all files.
    For every vertex, a bounding-cylinder pre-filter eliminates the vast
    majority of candidates before calling the expensive ``is_inside``
    method, making this efficient for millions of vertices.

    Parameters
    ----------
    gdml
        Path to the GDML file describing the L200 geometry.
    lh5_files
        A single LH5 file path **or** a sequence of paths. Each file must
        have fields ``xloc``, ``yloc``, ``zloc`` (in **metres**) under
        *vtx_group*.
    vtx_group
        HDF5 group name for the vertex data.
    save
        Whether to write the detector names back to each LH5 file.
    counts_yaml
        If given, write aggregate per-detector vertex counts to this
        YAML path.  The file will contain a flat
        ``{detector_name: n_vertices, ...}`` dictionary.
    return_evtids
        If True, also read and return the event IDs for each file.

    Returns
    -------
    NDArray | list[NDArray] | tuple[NDArray | list[NDArray], NDArray | list[NDArray]]
        If a single file is given, returns a single string array (and evtid array if requested).
        If a list of files is given, returns a list of string arrays
        (and a list of evtid arrays if requested).
    """
    det_map = build_detector_map(gdml)

    # Normalise to a list while remembering if the input was a single file.
    single_input = isinstance(lh5_files, str | Path)
    if single_input:
        lh5_files = [lh5_files]

    results: list[NDArray] = []
    evtids_list: list[NDArray] = []
    for i, lh5_file in enumerate(lh5_files):
        logger.debug("Processing file %d/%d: %s", i + 1, len(lh5_files), lh5_file)
        if return_evtids:
            det_names, evtids = _process_single_file(
                lh5_file, det_map, vtx_group, save=save, return_evtids=True
            )
            evtids_list.append(evtids)
        else:
            det_names = _process_single_file(lh5_file, det_map, vtx_group, save=save)
        results.append(det_names)

    # Write aggregate vertex counts to YAML if requested.
    if counts_yaml is not None:
        counts = aggregate_vertex_counts(
            results, evtid_arrays=evtids_list if return_evtids else None
        )
        counts_path = Path(counts_yaml)
        counts_path.parent.mkdir(parents=True, exist_ok=True)
        Props.write_to(counts_path, counts)
        logger.debug(
            "Wrote aggregate vertex counts (%d detectors) to %s",
            len(counts),
            counts_path,
        )

    if return_evtids:
        return (results[0], evtids_list[0]) if single_input else (results, evtids_list)
    return results[0] if single_input else results


def main() -> None:
    """CLI entry point for assigning HPGe detectors to LH5 vertices."""
    parser = argparse.ArgumentParser(
        description="Assign HPGe detector names to vertices stored in an LH5 file."
    )
    parser.add_argument(
        "--gdml",
        required=True,
        type=Path,
        help="Path to the L200 GDML geometry file.",
    )
    parser.add_argument(
        "--lh5-file",
        required=True,
        nargs="+",
        type=Path,
        help="One or more LH5 files containing vertex data.",
    )
    parser.add_argument(
        "--vtx-group",
        default="vtx",
        help="HDF5 group name for vertex data (default: 'vtx').",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write the 'det' field back to each LH5 file.",
    )
    parser.add_argument(
        "--counts-yaml",
        type=Path,
        default=None,
        help="Write aggregate per-detector vertex counts to this YAML file.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    assign_detectors_to_vertices(
        gdml=args.gdml,
        lh5_files=args.lh5_file,
        vtx_group=args.vtx_group,
        save=args.save,
        counts_yaml=args.counts_yaml,
    )


if __name__ == "__main__":
    main()

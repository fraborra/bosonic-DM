# Copyright (C) 2025 Francesco Borra
#

"""Check statistical uncertainties of dark compton detection efficiencies.

Usage
-----
After installing the package (``pixi run pip install -e .``), this module
is available as a CLI tool::

    pixi run check-dark-compton-stats
"""

from __future__ import annotations

import logging
import sys

# Mock tqdm.notebook before importing helper_lib to avoid Jupyter/ipywidgets ImportError in terminal execution
import tqdm

sys.modules["tqdm.notebook"] = tqdm

import polars as pl  # noqa: E402
from dbetto import Props  # noqa: E402
from legendmeta import LegendMetadata  # noqa: E402

from bosonic_dm.efficiency import compute_efficiency_from_lazyframe  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Calculate statistical uncertainties of detection efficiencies.

    This script computes the statistical uncertainties of the detection efficiencies
    for individual detectors using the simulation Parquet dataset.
    It is used to estimate how many additional events need to be generated
    to satisfy the relative statistical uncertainty requirement of <= 1%.
    """
    logger.info("Loading metadata...")
    version = "v2.1.5"
    base = f"/global/cfs/projectdirs/m2676/data/lngs/l200/public/prodenv/prod-blind/ref/{version}/"
    config = Props.read_from(base + "/config.json", subst_pathvar=True)["setups"][
        "l200"
    ]["paths"]

    meta = LegendMetadata(config["metadata"])
    timestamp = meta.dataprod.runinfo.p03.r000.phy.start_key
    chmap = meta.channelmap(timestamp)

    simulated_energies = list(range(200, 1100, 100))
    outdir_name = "data/v1/parquet/dark-compton"
    dictionaries_dir = "./data/v1/dictionaries"

    logger.info("Loading eres_dict...")
    eres_dict = Props.read_from(f"{dictionaries_dir}/eres_per_det_tot.yaml")

    logger.info("Loading parquet lazyframe...")
    lf = pl.scan_parquet(f"{outdir_name}", hive_partitioning=True)

    logger.info("Computing efficiencies...")
    ratio_dict = compute_efficiency_from_lazyframe(
        lf, eres_dict, simulated_energies, chmap
    )

    logger.info("\n--- RESULTS ---")
    # Analizziamo le incertezze per ciascuna energia simulata
    for ene in sorted(ratio_dict.keys()):
        logger.info("\nSimulated Energy: %d keV", ene)
        det_data = []
        for det_name, info in ratio_dict[ene].items():
            ratio = info["ratio"]
            ratio_sigma = info[
                "ratio_sigma"
            ]  # Incertezza bayesiana (deviazione standard a posteriori)
            n_primaries = info["n_primaries"]
            n_events = info["n_events"]

            # Incertezza statistica relativa (sigma_R / R)
            rel_sigma = ratio_sigma / ratio if ratio > 0 else 0.0

            # Identificazione del tipo di detector (BEGe, PPC, ICPC, COAX)
            det_type = "UNKNOWN"
            if det_name.startswith("B"):
                det_type = "BEGe"
            elif det_name.startswith("P"):
                det_type = "PPC"
            elif det_name.startswith("V"):
                det_type = "ICPC"
            elif det_name.startswith("C"):
                det_type = "COAX"

            det_data.append(
                {
                    "name": det_name,
                    "type": det_type,
                    "ratio": ratio,
                    "ratio_sigma": ratio_sigma,
                    "rel_sigma": rel_sigma,
                    "n_prim": n_primaries,
                    "n_evts": n_events,
                }
            )

        # Ordiniamo i detector in base all'incertezza relativa decrescente
        det_data_sorted_rel = sorted(
            det_data, key=lambda x: x["rel_sigma"], reverse=True
        )
        # Ordiniamo i detector in base all'incertezza assoluta decrescente
        det_data_sorted_abs = sorted(
            det_data, key=lambda x: x["ratio_sigma"], reverse=True
        )

        logger.info("Top 5 detectors with HIGHEST RELATIVE statistical uncertainty:")
        for d in det_data_sorted_rel[:5]:
            logger.info(
                "  %s (%s): Ratio = %.5f, Abs Unc = %.5f (%.3f%%), Rel Unc = %.5f (%.2f%%), n_prim = %d",
                d["name"],
                d["type"],
                d["ratio"],
                d["ratio_sigma"],
                d["ratio_sigma"] * 100,
                d["rel_sigma"],
                d["rel_sigma"] * 100,
                d["n_prim"],
            )

        logger.info("Top 5 detectors with HIGHEST ABSOLUTE statistical uncertainty:")
        for d in det_data_sorted_abs[:5]:
            logger.info(
                "  %s (%s): Ratio = %.5f, Abs Unc = %.5f (%.3f%%), Rel Unc = %.5f (%.2f%%), n_prim = %d",
                d["name"],
                d["type"],
                d["ratio"],
                d["ratio_sigma"],
                d["ratio_sigma"] * 100,
                d["rel_sigma"],
                d["rel_sigma"] * 100,
                d["n_prim"],
            )


if __name__ == "__main__":
    main()

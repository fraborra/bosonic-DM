# Copyright (C) 2025 Francesco Borra
#

"""Legacy functions kept for backward compatibility with scratch notebooks.

These functions are only used in ``do_smth.ipynb`` and other exploratory notebooks.
They are not part of the public API of ``bosonic_dm``.
"""

from __future__ import annotations

import awkward as ak
import matplotlib.pyplot as plt
from lgdo import lh5
from tqdm.notebook import tqdm


def select_channel(energies, channels, rawid):
    return energies[channels == rawid]


def prendi_ene_rawid(cvt_file):
    data_ak = lh5.read_as(
        "/evt", cvt_file, field_mask=["coincident", "geds", "trigger"], library="ak"
    )

    data_in_ge = data_ak[data_ak.coincident.geds]

    energies = ak.flatten(data_in_ge.geds.energy[data_in_ge.geds.multiplicity == 1])
    channels = ak.flatten(data_in_ge.geds.rawid[data_in_ge.geds.multiplicity == 1])

    return energies, channels


def riempi_dict(cvt_file, ges, chmap, stp_files, ene, thr):
    energies, channels = prendi_ene_rawid(cvt_file)

    det_ene = {}

    for ge in tqdm(ges):
        rawid = chmap[ge].daq.rawid
        stp_ge = lh5.read_as(f"/stp/{ge}", stp_files, library="ak")
        det_ene[ge] = {}
        det_ene[ge]["energy"] = energies[channels == rawid]
        det_ene[ge]["evtids"] = stp_ge.evtid
        tmp = det_ene[ge]["energy"]
        det_ene[ge]["ratio"] = len(tmp[(tmp < ene + thr) & (tmp > ene - thr)]) / len(
            det_ene[ge]["evtids"]
        )

    return det_ene


def get_n_primaries(ges, stp_files):
    det_prim = {}

    for ge in tqdm(ges):
        stp_ge = lh5.read_as(f"/stp/{ge}", stp_files, library="ak")
        det_prim[ge] = len(stp_ge.evtid)

    return det_prim


def compute_ratio(det_ene, ene, thr):
    for ge in det_ene.keys():
        tmp = det_ene[ge]["energy"]
        det_ene[ge]["ratio"] = len(tmp[(tmp < ene + thr) & (tmp > ene - thr)]) / len(
            det_ene[ge]["evtids"]
        )


def prendi_valori(det_ene):
    keys = list(det_ene.keys())

    values = []
    for ge in keys:
        values.append(det_ene[ge]["ratio"])

    return keys, values


def plot_e_det_type(ene_dict, ene, bins=300, lw=1):
    bege = ak.Array([])
    coax = ak.Array([])
    icpc = ak.Array([])
    ppc = ak.Array([])

    for ge in ene_dict.keys():
        if ge[0] == "B":
            bege = ak.concatenate([bege, ene_dict[ge]["energy"]])

        if ge[0] == "C":
            coax = ak.concatenate([coax, ene_dict[ge]["energy"]])

        if ge[0] == "V":
            icpc = ak.concatenate([icpc, ene_dict[ge]["energy"]])

        if ge[0] == "P":
            ppc = ak.concatenate([ppc, ene_dict[ge]["energy"]])

    plt.figure(figsize=(10, 6))
    plt.hist(bege, bins=bins, label="BEGe", histtype="step", linewidth=lw)
    plt.hist(ppc, bins=bins, label="PPC", histtype="step", linewidth=lw)
    plt.hist(coax, bins=bins, label="COAX", histtype="step", linewidth=lw)
    plt.hist(icpc, bins=bins, label="ICPC", histtype="step", linewidth=lw)
    plt.yscale("log")
    plt.legend(title=f"{ene}keV e-", fontsize=13)
    plt.xlabel("Processed Energy [keV]", fontsize=13)
    plt.savefig(f"notebooks/plots/det_type_energy_{ene}.png", dpi=300)
    plt.show()


def get_values_type(det_dict, ene, bins, lw=1):
    bege = ak.Array([])
    coax = ak.Array([])
    icpc = ak.Array([])
    ppc = ak.Array([])

    for ge in det_dict.keys():
        if ge[0] == "B":
            bege = ak.concatenate([bege, det_dict[ge]["energy"]])

        if ge[0] == "C":
            coax = ak.concatenate([coax, det_dict[ge]["energy"]])

        if ge[0] == "V":
            icpc = ak.concatenate([icpc, det_dict[ge]["energy"]])

        if ge[0] == "P":
            ppc = ak.concatenate([ppc, det_dict[ge]["energy"]])

    plt.figure(figsize=(10, 6))
    plt.hist(bege, bins=bins, label="BEGe", histtype="step", linewidth=lw)
    plt.hist(ppc, bins=bins, label="PPC", histtype="step", linewidth=lw)
    plt.hist(coax, bins=bins, label="COAX", histtype="step", linewidth=lw)
    plt.hist(icpc, bins=bins, label="ICPC", histtype="step", linewidth=lw)
    plt.yscale("log")
    plt.legend(title=f"{ene}keV e-", fontsize=13)
    plt.xlabel("Processed Energy [keV]", fontsize=13)
    plt.savefig(f"notebooks/plots/det_type_energy_{ene}.png", dpi=300)
    plt.show()

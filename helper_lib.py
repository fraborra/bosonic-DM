import numpy as np
import glob
import lgdo.lh5 as lh5
import os, json
import copy
import glob
import matplotlib.pyplot as plt
from pygama.pargen.utils import load_data
import pygama.math.distributions as pgf
from legendmeta import LegendMetadata
from dbetto import Props, TextDB, AttrsDict
import pandas as pd
from tqdm.notebook import tqdm
import awkward as ak

def select_channel(energies, channels, rawid):

    return energies[channels == rawid]


def prendi_ene_rawid(cvt_file):
    
    data_ak = lh5.read_as("/evt", cvt_file, field_mask=["coincident", "geds", "trigger"], library="ak")

    data_in_ge = data_ak[
        data_ak.coincident.geds
    ]
    
    energies = ak.flatten(data_in_ge.geds.energy[data_in_ge.geds.multiplicity == 1])
    channels = ak.flatten(data_in_ge.geds.rawid[data_in_ge.geds.multiplicity == 1])

    return energies, channels

def riempi_dict(cvt_file, ges, chmap, stp_files, ene, thr):

    energies, channels = prendi_ene_rawid(cvt_file)
    
    det_ene = {}
    
    for ge in tqdm(ges):
        rawid = chmap[ge].daq.rawid
        stp_ge = lh5.read_as(f"/stp/{ge}", stp_files, library='ak')
        det_ene[ge] = {}
        det_ene[ge]['energy'] = energies[channels == rawid]
        det_ene[ge]['evtids'] = stp_ge.evtid
        tmp = det_ene[ge]['energy']
        det_ene[ge]['ratio'] = len(tmp[(tmp<ene+thr) & (tmp >ene-thr)])/len(det_ene[ge]['evtids'])
    
    return det_ene

def get_n_primaries(ges, stp_files):
    
    det_prim = {}
    
    for ge in tqdm(ges):
        
        stp_ge = lh5.read_as(f"/stp/{ge}", stp_files, library='ak')
        det_prim[ge] = len(stp_ge.evtid)
    
    return det_prim


def compute_ratio(det_ene, ene, thr):
    for ge in det_ene.keys():
        tmp = det_ene[ge]['energy']
        det_ene[ge]['ratio'] = len(tmp[(tmp<ene+thr) & (tmp >ene-thr)])/len(det_ene[ge]['evtids'])

def prendi_valori(det_ene):
    keys = list(det_ene.keys())

    values = []
    for ge in keys:
        values.append(det_ene[ge]['ratio'])

    return keys, values

def plot_e_det_type(ene_dict, ene, bins = 300, lw = 1):

    B = ak.Array([])
    C = ak.Array([])
    I = ak.Array([])
    P = ak.Array([])
    
    for ge in ene_dict.keys():

        if ge[0]=="B":
            B = ak.concatenate([B, ene_dict[ge]['energy']])

        if ge[0]=="C":
            C = ak.concatenate([C, ene_dict[ge]['energy']])

        if ge[0]=="V":
            I = ak.concatenate([I, ene_dict[ge]['energy']])

        if ge[0]=="P":
            P = ak.concatenate([P, ene_dict[ge]['energy']])

    plt.figure(figsize=(10,6))
    plt.hist(B, bins = bins, label = 'BEGe', histtype='step', linewidth=lw)
    plt.hist(P, bins = bins, label = 'PPC', histtype='step', linewidth=lw)
    plt.hist(C, bins = bins, label = 'COAX', histtype='step', linewidth=lw)
    plt.hist(I, bins = bins, label = 'ICPC', histtype='step', linewidth=lw)
    plt.yscale('log')
    plt.legend(title=f'{ene}keV e-', fontsize = 13)
    plt.xlabel('Processed Energy [keV]', fontsize = 13)
    plt.savefig(f"notebooks/plots/det_type_energy_{ene}.png", dpi = 300)
    plt.show()

def get_values_type(det_dict):
    
    B = ak.Array([])
    C = ak.Array([])
    I = ak.Array([])
    P = ak.Array([])
    
    for ge in ene_dict.keys():

        if ge[0]=="B":
            B = ak.concatenate([B, ene_dict[ge]['energy']])

        if ge[0]=="C":
            C = ak.concatenate([C, ene_dict[ge]['energy']])

        if ge[0]=="V":
            I = ak.concatenate([I, ene_dict[ge]['energy']])

        if ge[0]=="P":
            P = ak.concatenate([P, ene_dict[ge]['energy']])

    plt.figure(figsize=(10,6))
    plt.hist(B, bins = bins, label = 'BEGe', histtype='step', linewidth=lw)
    plt.hist(P, bins = bins, label = 'PPC', histtype='step', linewidth=lw)
    plt.hist(C, bins = bins, label = 'COAX', histtype='step', linewidth=lw)
    plt.hist(I, bins = bins, label = 'ICPC', histtype='step', linewidth=lw)
    plt.yscale('log')
    plt.legend(title=f'{ene}keV e-', fontsize = 13)
    plt.xlabel('Processed Energy [keV]', fontsize = 13)
    plt.savefig(f"notebooks/plots/det_type_energy_{ene}.png", dpi = 300)
    plt.show()

def get_mean_fcc_det_type(ratio_dict):

    ratio_dict_means = {}
    
    for ene in ratio_dict.keys():
        ratio_dict_means[ene] = {}
        
        B = []
        C = []
        I = []
        P = []
        
        for ge in ratio_dict[ene].keys():

            if ge[0]=="B":
                B.append(ratio_dict[ene][ge]['ratio'])
            
            if ge[0]=="C":
                C.append(ratio_dict[ene][ge]['ratio'])
            
            if ge[0]=="V":
                I.append(ratio_dict[ene][ge]['ratio'])
            
            if ge[0]=="P":
                P.append(ratio_dict[ene][ge]['ratio'])

        ratio_dict_means[ene]['BEGe'] = np.mean(clean_array(B))
        ratio_dict_means[ene]['ICPC'] = np.mean(clean_array(I)) 
        ratio_dict_means[ene]['COAX'] = np.mean(clean_array(C)) 
        ratio_dict_means[ene]['PPC'] = np.mean(clean_array(P))

    return ratio_dict_means

def clean_array(arr):
    arr = np.asarray(arr)
    arr = arr.astype(float)
    clean = arr[(arr != 0) & np.isfinite(arr)]
    return clean


def ak_to_pandas(ak_obj1, ak_obj2):

    df = pd.DataFrame()
    
    df['energy']          = ak.to_numpy(ak.flatten(ak_obj1.energy))
    df['energy_sum']      = ak.to_numpy(ak_obj1.energy_sum)
    df['hit_idx']         = ak.to_numpy(ak.flatten(ak_obj1.hit_idx))
    df['is_good_channel'] = ak.to_numpy(ak.flatten(ak_obj1.is_good_channel))
    df['is_single_site']  = ak.to_numpy(ak.flatten(ak_obj1.is_single_site))
    df['multiplicity']    = ak.to_numpy(ak_obj1.multiplicity)
    df['rawid']           = ak.to_numpy(ak.flatten(ak_obj1.rawid))

    df['evtid'] = ak.to_numpy(ak_obj2.evtid)
    df['period'] = ak.to_numpy(ak_obj2.period)
    df['run'] = ak.to_numpy(ak_obj2.run)
    
    return df

    
def get_rawids_map(chmap, ges):
    
    rawids_map = {}

    for ge in ges:
        rawids_map[ge] = chmap[ge].daq.rawid

    return rawids_map

def get_values_sorted(det_dict, ges_sorted):
    values = []

    for ge in ges_sorted:
        values.append(det_dict[ge]['ratio'])

    return values, ges_sorted
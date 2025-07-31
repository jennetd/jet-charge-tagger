# Copyright (c) 2025 Komal Tauqeer
# Licensed under the MIT License. See LICENSE file for details.

import uproot
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
import awkward as ak

def root2df(sfile, tree_name, variables, **kwargs):
    f = uproot4.open(sfile)
    tree = f[tree_name]
    df = tree.arrays(variables, library="pd")
    return df

def pandas_to_hdf5(df, outdir, outfilename):
    df.to_hdf(outdir + '/'+ outfilename +'.h5', key='table', mode='w')

def save_dataset(dataset, outdir, outfilename):
    pandas_to_hdf5(dataset, outdir, outfilename)

def unstack_multi_df(df):
    df = df.unstack()
    #Remove subentries level and rename dataframe columns 
    df.columns = [a[0] + "_" +str(a[1]) for a in df.columns.to_flat_index()]
    return df

def merge_df(first, second, index_ignore=True):
    df = pd.concat((first, second), ignore_index=index_ignore)
    return df

def shuffle_df(dataset, random_state=42):
    return shuffle(dataset, random_state)

# Jennet replaces what was originally here
def prepare_input_dataset(sfile, samplename, treename="/Events"):

    events = NanoEventsFactory.from_root(
        {sfile:treename},
        schemaclass=PFNanoAODSchema,
        mode='eager',
        entry_stop=100
    ).events()

    jet0 = ak.firsts(events.FatJet)

    ak8_pf_candidates = events.FatJetPFCands
    
    jet0_constituents_indices = ak8_pf_candidates.pFCandsIdx[ak8_pf_candidates.jetIdx == 0]

    pf_candidates = events.PFCands[jet0_constituents_indices]

    first_115_pf = ak.pad_none(pf_candidates,115,clip=True)

    df_px = pd.DataFrame((first_115_pf.px).to_list(),columns=["PF_Px_"+str(i) for i in range(0,115)])
    df_py = pd.DataFrame((first_115_pf.py).to_list(),columns=["PF_Py_"+str(i) for i in range(0,115)])
    df_pz = pd.DataFrame((first_115_pf.pz).to_list(),columns=["PF_Pz_"+str(i) for i in range(0,115)])
    df_E  = pd.DataFrame((first_115_pf.E).to_list(),columns=["PF_E_"+str(i) for i in range(0,115)])
    df_q  = pd.DataFrame((first_115_pf.charge).to_list(),columns=["PF_q_"+str(i) for i in range(0,115)])
    df_w  = pd.DataFrame(events.Generator.weight,columns=["event_weight"])

    # Now get the truth label
    def getWplusBosons(genparticles):
        return genparticles[
            (genparticles.pdgId == 24)
            & genparticles.hasFlags(['fromHardProcess', 'isLastCopy'])
        ]

    def getWminusBosons(genparticles):
        return genparticles[
            (genparticles.pdgId == -24)
            & genparticles.hasFlags(['fromHardProcess', 'isLastCopy'])
        ]

    def getZBosons(genparticles):
        return genparticles[
            (genparticles.pdgId == 23)
            & genparticles.hasFlags(['fromHardProcess', 'isLastCopy'])
        ]
    
    # Positive
    wplusbosons = getWplusBosons(events.GenPart)
    dR_wplus = jet0.deltaR(wplusbosons)

    # Negative
    wminusbosons = getWminusBosons(events.GenPart)
    dR_wminus = jet0.deltaR(wminusbosons)

    # Zero
    zbosons = getZBosons(events.GenPart)
    dR_z = jet0.deltaR(zbosons)

    best_match = ak.argmin(ak.Array([dR_wplus,dR_z,dR_wminus]),axis=0)
    # Also allow for unmatched (3 - 1 -> 2 in the output df)
    best_match = ak.fill_none(ak.pad_none(best_match,1),3)

    label = ak.flatten(best_match - 1)
    df_label = pd.DataFrame(label,columns=["truth_label"])

    df = pd.concat([df_px,df_py,df_pz,df_E,df_q,df_w,df_label],axis=1)

    return df

    return dataset


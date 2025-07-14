# Copyright (c) 2025 Komal Tauqeer
# Licensed under the MIT License. See LICENSE file for details.

import uproot4
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

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

def prepare_input_dataset(sfile, treename, samplename, variables, labels, weights):
    dataset = root2df(sfile, treename, variables)
    dataset = unstack_multi_df(dataset)
    if samplename != 'ZJets':
        labels_data = root2df(sfile, treename, labels)
        dataset = dataset.join(labels_data) #Labels comes from the charge of lepton
        print(dataset)
    else:
        dataset[labels] = np.array(0.) #Add 0 label for Z jets
    weights_data = root2df(sfile, treename, weights)
    dataset = dataset.join(weights_data)

    return dataset


# Copyright (c) 2019 Huilin Qu
# Licensed under the MIT License. See LICENSE file for details.

import os
import optparse
import pandas as pd
import numpy as np
import awkward
#import uproot_methods
# Jennet replaces uproot_methods with coffea vector
from coffea.nanoevents.methods import vector
from sklearn.preprocessing import MultiLabelBinarizer
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def _transform(dataframe, mode, start=0, stop=-1, jet_size=0.8):
    from collections import OrderedDict
    v = OrderedDict()

    df = dataframe.iloc[start:stop]
    #def _col_list(prefix, max_particles=77):
    def _col_list(prefix, max_particles=70):
        return ['%s_%d'%(prefix,i) for i in range(max_particles)]

    _px = df[_col_list('PF_Px')].values
    _py = df[_col_list('PF_Py')].values
    _pz = df[_col_list('PF_Pz')].values
    _e = df[_col_list('PF_E')].values
    _q = df[_col_list('PF_q')].values

    mask = _e>0
    n_particles = np.sum(mask, axis=1)

    px = awkward.Array(_px[mask])
    py = awkward.Array(_py[mask])
    pz = awkward.Array(_pz[mask])
    energy = awkward.Array(_e[mask])
    charge = awkward.Array(_q[mask])

    p4 = awkward.zip(
        {
            "px": px,
            "py": py,
            "pz": pz,
            "E": energy,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )

#    .TLorentzVectorArray.from_cartesian(px, py, pz, energy)
    pt = p4.pt

    jet_p4 = p4

    # outputs
    old_label = df['truth_label']
    #if mode == "binary":
    #    new_label = [[1,0] if i == -1 else [0,1] for i in old_label]
    #    new_label = np.array(new_label)
    #    print (new_label)
    #    v['label'] = new_label
    #elif mode == "ternary":   
    #    new_label = []
    #    for i in old_label:
    #        if i == -1: new_label.append([1,0,0])
    #        if i == 1:new_label.append([0,1,0])
    #        if i == 0: new_label.append( [0,0,1])
    #    v['label'] = np.array(new_label)
    
    new_label = []
    for i in old_label:
        if i == -1: new_label.append([1,0,0])
        if i == 1:new_label.append([0,1,0])
        if i == 0: new_label.append( [0,0,1])
    
    v['label'] = np.array(new_label)

    v['event_weight'] = df['event_weight'].values
    v['jet_pt'] = jet_p4.pt
    v['jet_eta'] = jet_p4.eta
    v['jet_phi'] = jet_p4.phi
    v['jet_mass'] = jet_p4.mass
    v['n_parts'] = n_particles

    v['part_px'] = px
    v['part_py'] = py
    v['part_pz'] = pz
    v['part_energy'] = energy
    v['part_charge'] = charge

    v['part_pt_log'] = np.log(pt)
    v['part_ptrel'] = pt/v['jet_pt']
    v['part_logptrel'] = np.log(v['part_ptrel'])

    v['part_e_log'] = np.log(energy)
    v['part_erel'] = energy/jet_p4.energy
    v['part_logerel'] = np.log(v['part_erel'])

    v['part_raw_etarel'] = (p4.eta - v['jet_eta'])
    _jet_etasign = np.sign(v['jet_eta'])
#    _jet_etasign[_jet_etasign==0] = 1
    v['part_etarel'] = v['part_raw_etarel'] * _jet_etasign

    v['part_phirel'] = p4.delta_phi(jet_p4)
    v['part_deltaR'] = np.hypot(v['part_etarel'], v['part_phirel'])

    def _make_image(var_img, rec, n_pixels = 64, img_ranges = [[-0.8, 0.8], [-0.8, 0.8]]):
        wgt = rec[var_img]
        x = rec['part_etarel']
        y = rec['part_phirel']
        img = np.zeros(shape=(len(wgt), n_pixels, n_pixels))
        for i in range(len(wgt)):
            hist2d, xedges, yedges = np.histogram2d(x[i], y[i], bins=[n_pixels, n_pixels], range=img_ranges, weights=wgt[i])
            img[i] = hist2d
        return img

#     v['img'] = _make_image('part_ptrel', v)

    return v

def convert(source, destdir, basename, mode, step=None, limit=None):
    df = pd.read_hdf(source, key='table')
    logging.info('Total events: %s' % str(df.shape[0]))
    if limit is not None:
        df = df.iloc[0:limit]
        logging.info('Restricting to the first %s events:' % str(df.shape[0]))
    if step is None:
        step = df.shape[0]
    idx=-1
    while True:
        idx+=1
        start=idx*step
        if start>=df.shape[0]: break
        if not os.path.exists(destdir):
            os.makedirs(destdir)
        output = os.path.join(destdir, '%s_%d.awkd'%(basename, idx))
        logging.info(output)
        if os.path.exists(output):
            logging.warning('... file already exist: continue ...')
            continue
        v=_transform(df, mode, start=start, stop=start+step)
        awkward.to_parquet(v, output)



# ******************************************** Author: Komal Tauqeer *******************************************
# ***************************************** Date: July 11, 2025 ***********************************
# The jet charge tagger is trained for individual UL years, use UL18 as default year because it had the most statistics for the training
# The tagger gives output probabilites of a jet to belong to each class in the order W+, W-, and Z OR (+1, -1, 0) in terms of the electric charge

import os
import sys
sys.path.append("preprocessing/")
import numpy as np
import optparse
import tensorflow as tf
from tensorflow import keras
from load_datasets import Dataset
from array import array
from ROOT import *
import rootIO
from constants import *

parser = optparse.OptionParser()
parser.add_option("--year", dest="year", default= "UL18")
(options,args) = parser.parse_args()
year = options.year

def load_model():

    modelnumber = {"UL16preVFP": 30, "UL16postVFP": 30, "UL17": 30, "UL18": 29}
    modelpath = "/work/ktauqeer/ParticleNet/tf-keras/preprocessing/JetChargeTagger/ternary_training/{}/model_checkpoints/particle_net_lite_model.0{}.h5".format(year, modelnumber[year])
    model = keras.models.load_model(modelpath)

    return model

def predict_testset():
    
    #Load model
    model = load_model()
 
    #eval_path = "preprocessing/ternary_training/{y}/converted/<NameOfYourEvalFile>_0.awkd".format(y=year)
    eval_path = "preprocessing/ternary_training/{y}/converted/WpWnZ_test_{y}_0.awkd".format(y=year)
    print ("********************* Evaluating {} *****************************".format(eval_path))
    eval_dataset = Dataset(eval_path, data_format='channel_last')
    tagger_output= model.predict(eval_dataset.X)
    print (tagger_output)
    true_class = np.array([np.argmax(eval_dataset.y, axis=1)]).T
    predicted_class = np.array([np.argmax(tagger_output, axis=1)]).T
    nrows, ncolumns = np.shape(tagger_output)
    predicted_probabilites = []
    for row in range (nrows):
        predicted_probabilites.append(tagger_output[row][predicted_class[row]])

    #Store the predicted output probabilities as a branch back to the root files you started with
    #ofile = TFile.Open('ternary_training/{}/<NameOfYourEvalFile>_test.root'.format(year), "RECREATE")
    #tree = TTree("AnalysisTree", "AnalysisTree") #Replace the correct tree name
    #Wp = array('d', [0])
    #Wn = array('d', [0])
    #Z = array('d', [0])
    #Ind = array('d', [0])
    #trueInd = array('d', [0])
    #tree.Branch('jetchargetagger_prob_nodeWp', Wp, 'jetchargetagger_prob_nodeWp/D')
    #tree.Branch('jetchargetagger_prob_nodeWn', Wn, 'jetchargetagger_prob_nodeWn/D')
    #tree.Branch('jetchargetagger_prob_nodeZ', Z, 'jetchargetagger_prob_nodeZ/D')
    #tree.Branch('jetchargetagger_ind', Ind, 'jetchargetagger_ind/D')
    #tree.Branch('jetchargetagger_true_ind', trueInd, 'jetchargetagger_true_ind/D')
    #for itr in range(nrows):
    #    Wp[0] = tagger_output[itr,0]
    #    Wn[0] = tagger_output[itr,1]
    #    Z[0] = tagger_output[itr,2]
    #    Ind[0] = predicted_class.flatten()[itr]
    #    trueInd[0] = true_class.flatten()[itr]
    #    tree.Fill()

    #tree.Write()
    #ofile.Write()
    #ofile.Close()
    #print ("Branches added to the root file {}".format(ofile))

def main():

    global year
    predict_testset()

if __name__ == '__main__':
    main()

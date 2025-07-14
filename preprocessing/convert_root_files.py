# Copyright (c) 2025 Komal Tauqeer
# Licensed under the MIT License. See LICENSE file for details.

# Activate vir env: conda activate tf_py36 (the env yml file can be found here: /work/ktauqeer/jet-charge-tagger)

# Purpose: convert splitted root files to hdf5 format, then convert them into .awkd format while also calculating tagger input variables
# The root files must contains branches of Px, Py, Pz, E, q of the particle constiutents of the large-radius jets whose electric charge we want to predict using the tagger. The higher level input variables are automatically calculated on the go.

import os
import sys
import optparse
#local imports
from data_utils import *
from prepare_tagger_inputs import convert

inputvariables = ["PF_Px", "PF_Py", "PF_Pz", "PF_E", "PF_q"] #Puppi weights are applied already
eventWeights = {"TT": "event_weight"}
labels = ["lep_charge"]
treename = "AnalysisTree"
samples = ["TT"]

parser = optparse.OptionParser()
parser.add_option("--year", "--y", dest="year", help = "UL16preVFP, UL16postVFP, UL17, UL18", default= "UL18")
parser.add_option("--filepath", "--ifile", dest="ifile", help = "Give the path to the input root file", default= "/ceph/ktauqeer/ULNtuples/UL18/TTCR/jetchargeDP_note/TTCR_TTToSemiLeptonic_test.root")
(options,args) = parser.parse_args()

year = options.year
ifile = options.ifile

valid_years = ["UL16preVFP", "UL16postVFP", "UL17", "UL18"]
if year not in valid_years:
    raise ValueError(f"Invalid year: {year}. Must be one of {', '.join(valid_years)}.")

def prepare_testsets(year, outdir):
    opath = outdir + 'ternary_training/{}/Eval'.format(year) + '/'
    if not os.path.isdir(opath):
        os.makedirs(opath)

    for sample in samples:
        print ("***Converting {} file to pandas dataframe***".format(ifile))
        df = prepare_input_dataset(ifile, treename, sample, inputvariables, labels, eventWeights[sample])
        save_dataset(df, opath, "Eval_TTCR_{}_{}".format(sample, year))
        convert(os.path.join(opath, 'Eval_TTCR_{}_{}.h5'.format(sample,year)), destdir=opath+'/converted', basename='Eval_TTCR_{}_{}'.format(sample,year), mode="ternary")
        print ("****** Eval test files for \"{}\" are stored in \"{}\" *********".format(ifile,opath+'/converted'))


def main():
    outdir = './'
    prepare_testsets(year, outdir) #Adjust the input root file name accordingly using --ifile argument

if __name__ == "__main__":
    main()

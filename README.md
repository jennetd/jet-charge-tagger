# Jet Charge Tagger 

A toolkit to train and utilize a Dynamic Graph Convolutional Neural Network model named as the "jet charge tagger", to predict the electric charge of large-radius jets at the Large Hadron Collider. 

The details about the tagger can be found in the official publication from the CMS Collaboration along with the approved results on the [CERN CDS server](https://cds.cern.ch/record/2904357/files/DP2024_044.pdf). The tagger's model architecture is inspired by [ParticleNet](https://arxiv.org/abs/1902.08570).

-----
## Input requirements

A ROOT file containing exactly one jet per event. For training, each jet must have a true label corresponding to it's true electric charge.
For each jet, the file must have the Lorentz vector of particle constituents: Px, Py, Pz, E and electric charge q of the particle constituents, stored in separate branches. In addition, an activated conda environment is required, use the provided .yml to create the environment.

### Workflow

1. Split ROOT files

Split input ROOT files into training, validation, and test sets using:

```python preprocessing/split_rootfiles_manually.py```

This script preserves the original event ordering.

2. Convert and prepare input data

Convert the ROOT files and compute input features using:

```python preprocessing/convert_root_files.py```

This script performs the following:

Converts ROOT files to .h5, then converts .h5 to .awkd. It also computes derived features like jet\_pt, jet\_mass, etc.
The .awkd files will be saved in a directory such as:

```preprocessing/ternary_training/```

3. Run predictions

To classify jets using a trained model, run:

```python keras_predict_multi.py```

This will load a trained model from ternary_training/ and predict the probability for a jet to be:

W+ -like (charge +1)

W- -like (charge -1)

Z -like (neutral)

You can modify the script to store predictions back into ROOT files.
ROOT I/O utilities are included in the repository.

4. Retrain the model

To retrain the tagger with your own data use:

```python keras_train_multi.py```

The best model will be saved in the ternary_training/model_checkpoints.

Training curves will be saved as PDF files for visual inspection.


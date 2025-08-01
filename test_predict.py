from tensorflow import keras

def main():

#    modelpath = "./modelfiles/UL18/model_checkpoints/particle_net_lite_model.029.h5"
#    modelpath = "./modelfiles/UL17/model_checkpoints/particle_net_lite_model.030.h5"
    modelpath = "./modelfiles/UL16postVFP/model_checkpoints/particle_net_lite_model.030.h5"
#    modelpath = "./modelfiles/UL16preVFP/model_checkpoints/particle_net_lite_model.030.h5"
    
    model = keras.models.load_model(modelpath)

if __name__ == '__main__':
    main()

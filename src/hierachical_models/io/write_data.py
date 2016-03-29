__author__ = 'Julia Nitsch'

import h5py
import numpy as np
import os

def write_weights(file_name, weights,weight_layers, labels):
    """Writes the weights and labels to hdf5.

    @param file_name The filename of the database.
    @param weights NumPy array with weights of defined layer
    @param labals NumPy array containing the labels
    """

    # delete file if already exists
    try:
        os.remove(file_name)
    except OSError:
        pass

    # open file writeable
    file = h5py.File(file_name, "w")

    #create dataset for images
    dt_float32 = h5py.special_dtype(vlen=np.dtype('float32'))
    dset_activations = [None] * len(weight_layers)
    for layer_idx in range(0,len(weight_layers)):
        dset_activations[layer_idx] = file.create_dataset(weight_layers[layer_idx], data=weights[layer_idx])

    dset_label = file.create_dataset("/label", data=labels)

    file.close()
#!/usr/bin/env python
__author__ = 'jnitsch'

import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import os
import sys
caffe_root = os.environ['CAFFE']
sys.path.insert(0, caffe_root + 'python')
os.environ["GLOG_minloglevel"] = "2"
import caffe
os.environ["GLOG_minloglevel"] = "0"
import hierachical_models.io.load_data as io
import hierachical_models.io.write_data as io_write


def propagate(folder, kind_of_data, data, net, src_name, weight_layers):
    activations = [None] * len(weight_layers)
    percentage_printed = 0
    for img_idx in range(0,data.data.shape[0]):
        patch = data.data[img_idx, 0, :, :]
        net.blobs['data'].data[0,0,:,:] = patch
        out = net.forward()
        for layer_idx in range(0,len(weight_layers)):
            if img_idx == 0:
                activations[layer_idx] = np.array(net.blobs[weight_layers[layer_idx]].data)
            else:
                activations_current = np.array(net.blobs[weight_layers[layer_idx]].data)
                activations[layer_idx] = np.append(activations[layer_idx],activations_current,axis=0)

        percentage = round(float(img_idx)/float(data.data.shape[0]) * 100.0, 0)
        if (int(percentage) % 10 == 0) and (percentage != percentage_printed):
            print 'Completed %.2f %% of %s set %s' % (percentage,str(kind_of_data),str(src_name))
            percentage_printed = percentage
    directory = folder+'weights/'+str(src_name)+'/'+str(kind_of_data)+'/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory+src_name+'_weights.h5'
    io_write.write_weights(filename,activations,weight_layers,data.labels[:,0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    This script loads network and extracts weights from given layer and saves [weights labels] in top weight and data
    """, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--folder', help='top folder', default='')
    parser.add_argument('--weight_layers', help='name of layer whose weights should be stored', type=str)

    args = parser.parse_args()
    weight_layer = args.weight_layers
    weight_layers = []
    for layer in args.weight_layers.split(','):
        weight_layers.append(layer)
    folder = args.folder

    structure, nets = io.loadNetNames(folder+"net")
    weights = nets['combined']

    training, testing, sources = io.loadData(folder+"data", "data")

    #load caffe
    caffe.set_mode_cpu()
    net = caffe.Net(structure, weights, caffe.TEST)

    for src_name,training_data in training.items():
        # no need to create
        if src_name == 'combined':
            continue
        propagate(folder,'training',training_data,net,src_name,weight_layers)


    for src_name,testing_data in testing.items():
        # no need to create
        if src_name == 'combined':
            continue
        propagate(folder,'testing',testing_data,net,src_name,weight_layers)



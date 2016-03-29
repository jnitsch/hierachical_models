#!/usr/bin/env python
__author__ = 'jnitsch'

import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import sar_caffe_classifier.io.load_hdf5_data as io_in
import sar_caffe_classifier.io.write_hdf5_data as io_out
#load caffe env variable
import os
caffe_root = os.environ['CAFFE']

#add caffe to your sys path
import sys
sys.path.insert(0, caffe_root + 'python')

#import needed libs
import caffe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    This script loads network and extracts weights from given layer and saves [weights labels] in top weight and data
    """, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--folder', help='caffe net structure', default='')

    args = parser.parse_args()
    net_structure = args.net_structure
    weights = args.weights
    weight_layer = args.weight_layers
    weight_layers = []
    for layer in args.weight_layers.split(','):
        weight_layers.append(layer)
    filename = args.file_name
    filename_data = args.file_name_data

    #load caffe
    caffe.set_mode_cpu()
    net = caffe.Net(net_structure, weights, caffe.TEST)

    #load data and labels
    label = io_in.load_hdf5_labels(filename_data)
    img = io_in.load_hdf5_img(filename_data)

    if img.shape[1] is 1:
        print 'Images are grayscale'
    if img.shape[1] is 3:
        print 'Images are colorized -> PROBLEM'
        sys.exit()

    activations = [None] * len(weight_layers)
    for img_idx in range(0,img.shape[0]):
        patch = img[img_idx, 0, :, :]
        net.blobs['data'].data[0,0,:,:] = patch
        out = net.forward()
        for layer_idx in range(0,len(weight_layers)):
            if img_idx == 0:
                activations[layer_idx] = np.array(net.blobs[weight_layers[layer_idx]].data)
            else:
                activations_current = np.array(net.blobs[weight_layers[layer_idx]].data)
                activations[layer_idx] = np.append(activations[layer_idx],activations_current,axis=0)
        if img_idx % 500 == 0:
            print 'Completed %.2f %%' % round(float(img_idx)/float(img.shape[0]) * 100.0, 2)

    for layer_idx in range(0,len(weight_layers)):
        print 'weight shape: ',
        print activations[layer_idx].shape
    print 'label shape: ',
    print label.shape

    io_out.write_weights(filename,activations,weight_layers,label)




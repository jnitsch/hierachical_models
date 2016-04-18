#!/usr/bin/env python
___author__ = 'Julia Nitsch'
import argparse
import argparse
from argparse import RawTextHelpFormatter
import hierachical_models.io.load_data as io
import numpy as np
import hierachical_models.metrics.classifier_metrics as metrics
import sys
import os
caffe_root = os.environ['CAFFE']
sys.path.insert(0, caffe_root + 'python')
os.environ["GLOG_minloglevel"] = "2"
import caffe
os.environ["GLOG_minloglevel"] = "0"


def main():
    parser = argparse.ArgumentParser(description='''
    This script test different classifiers
    ''', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--folder', help='top folder of project (needs to contain a data and net folder)', default='')

    args = parser.parse_args()
    folder = args.folder

    print "load nets ..."
    structure, nets = io.loadNetNames(folder+"net")

    training, testing, sources = io.loadData(folder+"data", "data")

    n_classes = int(max(training['combined'].labels) + 1)
    print 'Found '+str(n_classes)+' classes'

    for testing_source in sources:
        print 'Found source '+str(testing_source)

    accuracy = {}
    mae = {}
    auc = {}
    for training_source,net in nets.items():
        caffe.set_mode_cpu()
        net = caffe.Net(structure, net, caffe.TEST)
        #test all combinations
        for testing_source in sources:
            test_data = testing[testing_source].data
            test_labels = testing[testing_source].labels
            y_prob = [None] * test_data.shape[0]
            y_labels = [None] * test_data.shape[0]
            for idx in range(0,test_data.shape[0]):
                patch = test_data[idx, 0, :, :]
                net.blobs['data'].data[0,0,:,:] = patch
                out = net.forward()
                y_prob[idx] = np.array(out['prob'][0])
                y_labels[idx] = np.argmax(out['prob'][0])

            y_prob = np.asarray(y_prob)
            accuracy[training_source+'_'+testing_source] = metrics.accuracy(test_labels, y_labels)
            mae[training_source+'_'+testing_source] = metrics.mae(test_labels, y_prob, n_classes)
            auc[training_source+'_'+testing_source] = metrics.AUC(test_labels, y_prob, n_classes)
            print 'Training from source: '+str(training_source)+' Tested from source: '+str(testing_source)+' ',
            print 'acc: ',
            print accuracy[training_source+'_'+testing_source],
            print ' - mae: ',
            print mae[training_source+'_'+testing_source],
            print ' - auc: ',
            print auc[training_source+'_'+testing_source]




if __name__ == "__main__":
    main()
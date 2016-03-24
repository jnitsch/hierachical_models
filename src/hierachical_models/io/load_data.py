#!/usr/bin/env python
___author__ = 'Julia Nitsch'

from collections import namedtuple
import h5py
import numpy as np
import os

Dataset=namedtuple("Dataset", ["data", "labels"])

def readh5(filename, datalayer):
    file_id = h5py.File(filename, 'r')
    data = np.array(file_id[datalayer])
    labels = np.array(file_id['/label'])
    return Dataset(data,labels)

def loadData(folder):
    sources = 0

    #sort it alphabetically to know the order of the sources
    directories = os.listdir(folder)
    directories = sorted(directories)
    tr = {}
    te = {}
    for directory_name in directories:
        if os.path.isdir(os.path.join(folder, directory_name)):
            print directory_name
            dir_path = os.path.join(folder, directory_name)

            for kind_of_data in os.listdir(dir_path):
                print '  ' + kind_of_data
                path_to_data = os.path.join(dir_path, kind_of_data)
                loaded_data = []
                for file in os.listdir(path_to_data):
                    print '    ' + file
                    file = os.path.join(path_to_data, file)
                    if len(loaded_data) == 0:
                        loaded_data = readh5(file, "ip1")
                    else:
                        new_data = readh5(file, "ip1")
                        loaded_data = Dataset(np.concatenate((loaded_data.data,new_data.data)),np.concatenate((loaded_data.labels,new_data.labels)))
                if kind_of_data == "training":
                    tr["d"+str(sources)] = loaded_data
                elif kind_of_data == "testing":
                    te["d"+str(sources)] = loaded_data
                else:
                    print 'Error in loading!'
        sources += 1

    combined_tr = []
    combined_te = []
    for src_idx in range(sources):
        if src_idx == 0:
            combined_tr = tr['d'+str(src_idx)]
            combined_te = te['d'+str(src_idx)]
        else:
            combined_tr = Dataset(np.concatenate((combined_tr.data,tr['d'+str(src_idx)].data)),np.concatenate((combined_tr.labels,tr['d'+str(src_idx)].labels)))
            combined_te = Dataset(np.concatenate((combined_te.data,te['d'+str(src_idx)].data)),np.concatenate((combined_te.labels,te['d'+str(src_idx)].labels)))
    tr['combined'] = combined_tr
    te['combined'] = combined_te
    return tr, te, sources
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


def loadData(folder, layer):
    sources = 0
    name_of_sources = []
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
                        loaded_data = readh5(file, layer)
                    else:
                        new_data = readh5(file, layer)
                        loaded_data = Dataset(np.concatenate((loaded_data.data,new_data.data)),np.concatenate((loaded_data.labels,new_data.labels)))
                if kind_of_data == "training":
                    tr[directory_name] = loaded_data
                elif kind_of_data == "testing":
                    te[directory_name] = loaded_data
                else:
                    print 'Error in loading!'
        name_of_sources.append(directory_name)
        sources += 1

    combined_tr = []
    combined_te = []
    src_idx = 0
    for src in name_of_sources:
        if src_idx == 0:
            combined_tr = tr[src]
            combined_te = te[src]
        else:
            combined_tr = Dataset(np.concatenate((combined_tr.data,tr[src].data)),np.concatenate((combined_tr.labels,tr[src].labels)))
            combined_te = Dataset(np.concatenate((combined_te.data,te[src].data)),np.concatenate((combined_te.labels,te[src].labels)))
        src_idx += 1

    tr['combined'] = combined_tr
    te['combined'] = combined_te
    name_of_sources.append('combined')
    return tr, te, name_of_sources

def loadNetNames(folder):
    directories = os.listdir(folder)
    directories = sorted(directories)
    net_structure = str()
    nets = {}
    for directory_name in directories:
        if os.path.isdir(os.path.join(folder, directory_name)):
            print directory_name
            dir_path = os.path.join(folder, directory_name)

            for net in os.listdir(dir_path):
                print net
                nets[directory_name] = os.path.join(dir_path, net)
        else:
            #found net structure
            print 'Net structure '+directory_name
            net_structure = os.path.join(folder, directory_name)

    return net_structure, nets
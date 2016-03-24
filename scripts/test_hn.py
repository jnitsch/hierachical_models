#!/usr/bin/env python
___author__ = 'Julia Nitsch'
import argparse
from argparse import RawTextHelpFormatter
import hierachical_models.io.load_data as io
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='''
    This script test different classifiers
    ''', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--n_classes', help='amount of classes', default='')
    parser.add_argument('--folder', help='folder where different classes are stored', default='')

    args = parser.parse_args()
    n_classes = int(args.n_classes)
    folder = args.folder

    training, testing, n_sources = io.loadData(folder)

    print 'Label training d1[0] %d, training d2[0] %d, training combined[0] %d' % (training['d0'].labels[0], training['d1'].labels[0], training['combined'].labels[0])

    print 'Finished!'

if __name__ == "__main__":
    main()

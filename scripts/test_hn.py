#!/usr/bin/env python
___author__ = 'Julia Nitsch'
import argparse
from argparse import RawTextHelpFormatter
from collections import namedtuple
import hierachical_models.metrics.classifier_metrics as metrics
import hierachical_models.io.load_data as io
import numpy as np
from sklearn import linear_model
from sklearn import naive_bayes


def pprinttable(rows):
  if len(rows) > 1:
    headers = rows[0]._fields
    lens = []
    for i in range(len(rows[0])):
      lens.append(len(max([x[i] for x in rows] + [headers[i]],key=lambda x:len(str(x)))))
    formats = []
    hformats = []
    for i in range(len(rows[0])):
      if isinstance(rows[0][i], int):
        formats.append("%%%dd" % lens[i])
      else:
        formats.append("%%-%ds" % lens[i])
      hformats.append("%%-%ds" % lens[i])
    pattern = " | ".join(formats)
    hpattern = " | ".join(hformats)
    separator = "-+-".join(['-' * n for n in lens])
    print hpattern % tuple(headers)
    print separator
    _u = lambda t: t.decode('UTF-8', 'replace') if isinstance(t, str) else t
    for line in rows:
        print pattern % tuple(_u(t) for t in line)
  elif len(rows) == 1:
    row = rows[0]
    hwidth = len(max(row._fields,key=lambda x: len(x)))
    for i in range(len(row)):
      print "%*s = %s" % (hwidth,row._fields[i],row[i])


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

    clfs = {"log":linear_model.LogisticRegression(),
            "bayes":naive_bayes.GaussianNB()}

    tr = []
    te = []
    accuracy = {}
    mae = {}
    auc = {}
    for trn,trd in training.items():
        for clfn,clf in clfs.items():
            clf.fit(trd.data,trd.labels)
            print clfn+': '+trn+' ',
            for ten,ted in testing.items():
                print ten+' ',
                accuracy[trn+'_'+ten+'_'+clfn] = metrics.accuracy(ted.labels, clf.predict(ted.data))
                predicted_probs = clf.predict_proba(ted.data)
                mae[trn+'_'+ten+'_'+clfn] = metrics.mae(ted.labels, predicted_probs, n_classes)
                auc[trn+'_'+ten+'_'+clfn] = metrics.AUC(ted.labels, predicted_probs, n_classes)
                print 'acc: ',
                print accuracy[trn+'_'+ten+'_'+clfn],
                print ' - mae: ',
                print mae[trn+'_'+ten+'_'+clfn],
                print ' - auc: ',
                print auc[trn+'_'+ten+'_'+clfn]








if __name__ == "__main__":
    main()

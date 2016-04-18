#!/usr/bin/env python
___author__ = 'Julia Nitsch'

import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from scipy import interp

def testProbVector(n_classes, test_labels):
    y_test = [None] * len(test_labels)
    for i in range(0,len(test_labels)):
        prob = np.array([0.0] * n_classes)
        prob[int(test_labels[i])] = 1
        y_test[i] = prob
    y_test = np.asarray(y_test)
    return y_test


def accuracy(test_labels, predicted_labels):
    return metrics.accuracy_score(test_labels,predicted_labels)


def mae(test_labels, predicted_probabilities, n_classes):
    y_test = testProbVector(n_classes, test_labels)
    return metrics.mean_absolute_error(y_test,predicted_probabilities)


def AUC(test_labels, predicted_labels, n_classes):
    y_test = testProbVector(n_classes, test_labels)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(0,n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:,i], predicted_labels[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), predicted_labels.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return np.asarray(roc_auc)
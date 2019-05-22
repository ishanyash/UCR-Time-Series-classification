# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:56:46 2019

@author: Ishan Yash
"""


import numpy as np
import pyts
from pyts.classification import KNeighborsClassifier


#print("pyts: {0}".format(pyts.__version__))

PATH = "G:/Coding/ML/UCRArchive_2018/" # Change this value if necessary

dataset_list = ["CBF", "ECG200", "GunPoint", "MiddlePhalanxTW", "Plane", "SyntheticControl"]

warping_window_list = [0.03, 0., 0., 0.03, 0.05, 0.06]


error_ed_list = []
error_dtw_list = []
error_dtw_w_list = []
#default_rate_list = []

for i, (dataset, warping_window) in enumerate(zip(dataset_list, warping_window_list)):
    print("Dataset: {}".format(dataset))
    
    file_train = PATH + str(dataset) + "/" + str(dataset) + "_TRAIN.tsv"
    file_test = PATH + str(dataset) + "/" + str(dataset) + "_TEST.tsv"
    
    train = np.genfromtxt(fname=file_train, delimiter="\t", skip_header=0)
    test = np.genfromtxt(fname=file_test, delimiter="\t", skip_header=0)

    X_train, y_train = train[:, 1:], train[:, 0]
    X_test, y_test = test[:, 1:], test[:, 0]

    clf_ed = KNeighborsClassifier(metric='euclidean')
    clf_dtw = KNeighborsClassifier(metric='dtw')
    clf_dtw_w = KNeighborsClassifier(metric='dtw_sakoechiba',
                                     metric_params={'window_size': warping_window})

    # Euclidean Distance
    error_ed = 1 - clf_ed.fit(X_train, y_train).score(X_test, y_test)
    print("Error rate with Euclidean Distance: {0:.4f}".format(error_ed))
    error_ed_list.append(error_ed)
    
    # Dynamic Time Warping
    error_dtw = 1 - clf_dtw.fit(X_train, y_train).score(X_test, y_test)
    print("Error rate with Dynamic Time Warping: {0:.4f}".format(error_dtw))
    error_dtw_list.append(error_dtw)
    
    # Dynamic Time Warping with a learned warping window
    error_dtw_w = 1- clf_dtw_w.fit(X_train, y_train).score(X_test, y_test)
    print("Error rate with Dynamic Time Warping with a learned warping "
          "window: {0:.4f}".format(error_dtw_w))
    error_dtw_w_list.append(error_dtw_w)
    
    print()
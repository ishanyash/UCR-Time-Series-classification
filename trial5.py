# -*- coding: utf-8 -*-
"""
Created on Tue May 14 17:24:02 2019

@author: Ishan Yash
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

full_train_data = pd.read_csv('CBF_TRAIN.txt', header = None)
full_test_data = pd.read_csv('CBF_TEST.txt', header = None)

train_dataset = full_train_data.iloc[:,range(1,full_train_data.shape[1])] #traindatawithoutlabel
label_train = full_train_data.iloc[:,0] #trainlabel

test_dataset = full_test_data.iloc[:,range(1,full_test_data.shape[1])] #testdatawithoutlabel
label_test = full_test_data.iloc[:,0] #testlabel

train_dataset = train_dataset.values
label_train = label_train.values

test_dataset=test_dataset.values
label_test=label_test.values


#Classification_algorithm

def class_algo(train_dataset,label_train,classification_obj):
    best_case = float('inf')
    for i in range(1,len(label_train)):
        global predicted_class
        compare_the_obj = train_dataset[i,:]
        #a = compare_the_obj.reshape(1,-1)
        #b = classification_obj.reshape(1,-1)
        dist = euclidean_distances(compare_the_obj,classification_obj)
        if dist < best_case:
            predicted_class = label_train[i]
        best_case = dist

correct = 0
for j in range(1,len(label_test)):
    classification_obj = test_dataset[j,:]
    actual_class = label_test[j]
    predicted_class = class_algo(train_dataset,label_train,classification_obj)
    if predicted_class == actual_class:
        correct +=1
print('Predicted class: ',predicted_class,'actual class: ', actual_class)
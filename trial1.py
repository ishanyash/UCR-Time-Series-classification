# -*- coding: utf-8 -*-
"""
Created on Wed May 15 01:35:31 2019

@author: Ishan Yash
"""

import numpy as np
import pandas as pd


full_train_data = pd.read_csv('ECG200_TRAIN.txt', header = None)
full_test_data = pd.read_csv('ECG200_TEST.txt', header = None)

train_dataset = full_train_data.iloc[:,range(1,full_train_data.shape[1])] #traindatawithoutlabel
label_train = full_train_data.iloc[:,0] #trainlabel

test_dataset = full_test_data.iloc[:,range(1,full_test_data.shape[1])] #testdatawithoutlabel
label_test = full_test_data.iloc[:,0] #testlabel

train_dataset = train_dataset.values
label_train = label_train.values

test_dataset=test_dataset.values
label_test=label_test.values

correct = 0
        
#Classification_algorithm

def class_algo(train_dataset,label_train,classification_obj):
    best_case = float('inf')
    for i in range(1,len(label_test)):
        global predicted_class,compare_the_obj
        classification_obj = test_dataset[i,:]
        classification_obj = classification_obj.reshape(1,-1)
        for j in range(1,len(label_train)):
            compare_the_obj = train_dataset[j,:]
            compare_the_obj = compare_the_obj.reshape(1,-1)
            
            #dist = euclidean_distances(compare_the_obj,classification_obj)
            #dist = np.sqrt(np.sum((np.array(compare_the_obj)-np.array(classification_obj))**2))
            dist = np.linalg.norm(np.array(compare_the_obj)-np.array(classification_obj))
            if dist < best_case:
                predicted_class = label_train[j]
                best_case = dist
    return 

for k in range(1,len(label_test)):
    classification_obj_1 = test_dataset[k,:]
    actual_class = label_test[k]
    predicted_class = class_algo(train_dataset,label_train,classification_obj_1)
    if predicted_class == actual_class:
        correct = correct + 1
        

        
        
#print("Predicted class: {0} ".format(correct))
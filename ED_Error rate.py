# -*- coding: utf-8 -*-
"""
Created on Wed May 15 01:35:31 2019

@author: Ishan Yash
"""

import numpy as np
import pandas as pd
import xlsxwriter
import numpy as np
from saxpy.znorm import znorm
from saxpy.sax import ts_to_string

#from sklearn.metrics.pairwise import euclidean_distances

full_train_data = pd.read_csv('Coffee_TRAIN', header = None)
full_test_data = pd.read_csv('Coffee_TEST', header = None)

train_dataset = full_train_data.iloc[:,range(1,full_train_data.shape[1])] #traindatawithoutlabel
label_train = full_train_data.iloc[:,0] #trainlabel

test_dataset = full_test_data.iloc[:,range(1,full_test_data.shape[1])] #testdatawithoutlabel
label_test = full_test_data.iloc[:,0] #testlabel
train_dataset = train_dataset.values
label_train = label_train.values

test_dataset=test_dataset.values
label_test=label_test.values

'''
test_dataset = shuffle(test_dataset)
label_test = shuffle(label_test)
train_dataset = shuffle(train_dataset)
label_train = shuffle(label_train)
'''
#Classification_algorithm
#best_case = 0.0
def euclidean(a, b):
    return np.sqrt(np.sum((np.array(a)-np.array(b))**2))

count=0
def class_algo(traindata,trainlabel,classobj):
    best_case = float("inf")
    predicted_class =0
    global count
    count+=1
    for j in range(-1,len(trainlabel)):
        compare_the_obj = traindata[j,:]
        #compare_the_obj = compare_the_obj.reshape(1,-1)
        dist = euclidean(compare_the_obj,classobj)
        #dist = np.sqrt(np.sum((np.array(compare_the_obj)-np.array(classobj))**2))
        #dist = euclidean_distances(compare_the_obj,classobj)
        #dist = np.linalg.norm(np.array(compare_the_obj)-np.array(classobj))
        if dist < best_case:
            predicted_class = trainlabel[j]
            best_case = dist
    #w = best_case.readline()
    return predicted_class 


correct = 0

for l in range(1,len(label_test)):
    classification_obj_1 = test_dataset[l,:]
    #classification_obj_1 = classification_obj_1.reshape(1,-1)
    actual_class = label_test[l]
    predicted_class = class_algo(train_dataset,label_train,classification_obj_1)
    if predicted_class == actual_class:
        correct =correct+1
        
#print('ED',sum(w)/len(trainlabel))
print("Correct", correct, 'Count' , count)
print('ED error rate: ', (count-correct)/count)



    



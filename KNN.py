#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 09:53:18 2017

@author: liubing
"""
import time
import input_data
#import pickle
import numpy as np
import scipy.io as sio
import time
start_time=time.time()
num_labeled=5

dataSet = input_data.read_data_sets('I_feature.h5',num_labeled)
train_data=dataSet.train.images
train_label=dataSet.train.labels
test_data=dataSet.test.images
test_label=dataSet.test.labels
print 'train.......'
from sklearn import neighbors
clf=neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(train_data,train_label)
print('training score:%f'%clf.score(train_data,train_label))
print('testing score:%f'%clf.score(test_data,test_label))
end_time=time.time()
print(end_time-start_time)


label_img=clf.predict(test_data)

import pickle
f=open('./gt_I.data','rb')
gt_index=pickle.load(f)
f.close()
m=145
n=145
final_result=np.zeros((m,n),dtype=np.int8)
final_result=final_result.reshape(m*n)
final_result[gt_index]=label_img+1
final_result=final_result.reshape(m,n)
sio.savemat('I-25.mat',{'final_result':final_result})

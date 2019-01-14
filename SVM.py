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

num_labeled=5

dataSet = input_data.read_data_sets('I_feature.h5',num_labeled)
train_data=dataSet.train.images
train_label=dataSet.train.labels
test_data=dataSet.test.images
test_label=dataSet.test.labels
print 'train.......'
#svm classification
from sklearn import svm
from sklearn import cross_validation

skf = cross_validation.StratifiedKFold(train_label,n_folds=4)

start=time.time()

max_ac=[]
cc=[]
gg=[]
ac=[]
for skf1,skf2 in skf:
    train_x=train_data[skf1]
    train_y=train_label[skf1]
    test_x=train_data[skf2]
    test_y=train_label[skf2]
    C=np.logspace(-2,8,11,base=2)
    gamma=np.logspace(-2,8,11,base=2)
    max_accuracy=0
    best_c=0
    best_g=0
    for i in C:
        for j in gamma:
            svr=svm.SVC(decision_function_shape='ovo',kernel='rbf',C=i,gamma=j)
            svr.fit(train_x,train_y)
            accuracy=svr.score(test_x,test_y)
            print 'max_accuracy:',max_accuracy,'.....accuracy_test:',accuracy#,'.....accuracy_train:',svr.score(train_data,train_label)
            if accuracy>max_accuracy:
                max_accuracy=accuracy
                best_c=i
                best_g=j
    max_ac.append(max_accuracy)
    cc.append(best_c)
    gg.append(best_g)
    #svr=svm.SVC(decision_function_shape='ovo',kernel='rbf',C=best_c,gamma=best_g)
    #svr.fit(train_data,train_label)
    #accuracy=svr.score(test_data,test_label)
    #ac.append(accuracy)

max_temp=0
index=0
for i in range(4):
    if max_ac[i]>max_temp:
        index=i
svr=svm.SVC(decision_function_shape='ovo',kernel='rbf',C=cc[index],gamma=gg[index])
svr.fit(train_data,train_label)
accuracy=svr.score(test_data,test_label)
print(accuracy)
end=time.time()
print('parameters time:'+str(end-start))
'''
label_img=svr.predict(test_data)

import pickle
f=open('./gt_I.data','rb')
gt_index=pickle.load(f)
f.close()

final_result=np.zeros((145,145),dtype=np.int8)
final_result=final_result.reshape(145*145)
final_result[gt_index]=label_img+1
final_result=final_result.reshape(145,145)

import matplotlib.pyplot as plt
#import matplotlib
cmap=plt.get_cmap('jet', 17)
fig=plt.figure()
pp=plt.imshow(final_result,cmap=cmap)
fig.savefig('IP-SVM.eps')
'''
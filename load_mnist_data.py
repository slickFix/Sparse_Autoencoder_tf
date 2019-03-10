#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 17:53:05 2019

@author: siddharth
"""

import numpy as np
import os

project_dir = '/home/siddharth/workspace-python/Sparse_Autoencoder_tf'
os.chdir(project_dir)

dataset_path  = './data/mnist'

def subset_fn(tr_x,tr_y,te_x,te_y,digit_range):
    
    # creating an array of the required digits
    subset_label = np.arange(digit_range[0],digit_range[1]+1)
    train_data_sub = []
    train_label_sub = []
    test_data_sub = []
    test_label_sub = []
    
    for i in subset_label:
        
        # returns a tuple whose 1st elements is the required array 
        train_sub_idx = np.where(tr_y==i)
        test_sub_idx = np.where(te_y==i)
        A = tr_x[train_sub_idx[0],:]
        C = te_x[test_sub_idx[0],:]

        B = tr_y[train_sub_idx[0]]
        D = te_y[test_sub_idx[0]]
			
        train_data_sub.append(A)
        train_label_sub.append(B)
        test_data_sub.append(C)
        test_label_sub.append(D)
    
    # finally creating the list of the required digits
    tr_x = train_data_sub[0]
    tr_y = train_label_sub[0]
    te_x = test_data_sub[0]
    te_y = test_label_sub[0]
    
    for i in range(digit_range[1]-digit_range[0]):
        tr_x = np.concatenate((tr_x,train_data_sub[i+1]),axis=0)
        tr_y = np.concatenate((tr_y,train_label_sub[i+1]),axis=0)
        te_x = np.concatenate((te_x,test_data_sub[i+1]),axis=0)
        te_y = np.concatenate((te_y,test_label_sub[i+1]),axis=0)
     
    return tr_x,tr_y,te_x,te_y

    
def mnist(ntrain=60000,ntest=10000,subset=True,digit_range=[0,2],shuffle=True):
    
    # loading data
    fd  = open(os.path.join(dataset_path,'train-images-idx3-ubyte'))
    loaded = np.fromfile(fd,dtype=np.uint8)
    tr_x = loaded[16:].reshape((60000,28*28)).astype('float')
    
    fd = open(os.path.join(dataset_path,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(fd,dtype=np.uint8)
    tr_y = loaded[8:].reshape((60000)).astype(float)
    
    fd = open(os.path.join(dataset_path,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(fd,dtype=np.uint8)
    te_x = loaded[16:].reshape((10000,28*28)).astype(float)

    fd = open(os.path.join(dataset_path,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(fd,dtype=np.uint8)
    te_y = loaded[8:].reshape((10000)).astype(float)
    
    
    # scaling image
    tr_x = tr_x/255.
    te_x = te_x/255.
    
    # providing the required number of train and test record
    tr_x = tr_x[:ntrain]
    tr_y = tr_y[:ntrain]
    
    te_x = te_x[:ntest]
    te_y = te_y[:ntest]
    
    tr_y = np.array(tr_y)
    te_y = np.array(te_y)
    
    
    # if the required digits are subset of all the digits present in the actual DB
    if subset:
        tr_x,tr_y,te_x,te_y = subset_fn(tr_x,tr_y,te_x,te_y,digit_range)
            

            
    return tr_x,tr_y,te_x,te_y

if __name__ == '__main__':
    tr_x,tr_y,te_x,te_y = mnist()
    
    
    
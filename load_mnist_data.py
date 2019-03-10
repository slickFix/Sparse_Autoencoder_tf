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
    
    
    
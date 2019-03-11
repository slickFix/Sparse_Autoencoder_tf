#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 17:38:45 2019

@author: siddharth
"""

import numpy as np
import tensorflow as tf

from load_mnist_data import mnist


if __name__ == '__main__':
    
    # LOADING DATA
    digit_range = [0,9] # inclusive of both the values
    
    tr_x,tr_y,te_x,te_y = mnist(ntrain=6000,ntest=1000,digit_range=digit_range)
    
    # making the count of all the the digit class to 100
    
    digit_count = [0]*(digit_range[1]-digit_range[0]+1)
    req_index = []
    
    for i in range(tr_x.shape[0]):
        if digit_count[int(tr_y[i])] == 100:
            continue
        else:
            digit_count[int(tr_y[i])]+=1
            req_index.append(i)
    
    tr_x = tr_x.take(req_index,axis = 0) # doing the subset of the 6k records
    tr_y = tr_y.take(req_index,axis = 0) # doing the subset of the 6k records
    
    tr_y = tr_y.reshape(-1,1) # reshaping from (1000,) to (1000,1)
    te_y = te_y.reshape(-1,1) # reshaping from (1000,) to (1000,1)
    
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 17:38:45 2019

@author: siddharth
"""

import numpy as np
import tensorflow as tf

from load_mnist_data import mnist

def create_placeholder(n_x):
    
    x_ph = tf.placeholder(tf.float32,shape=[None,n_x],name='X_ph')
    y_ph = tf.placeholder(tf.int64,shape=[None],name = 'Y_ph') # using 1 instead of n_y as we will use sparse_softmax_crossentropy
    
    return x_ph,y_ph

def initialise_parameter(n_x,n_y):
    
    # tf essentials
    wt_init = tf.variance_scaling_initializer()
    tf.set_random_seed(1)
    
    neurons_h1 = 200
    n_outputs = int(n_y)
    
    # fully connected weights and bias
    w1_fc = tf.Variable(wt_init([n_x,neurons_h1]),dtype = tf.float32,name = 'w1_fc')
    w2_fc = tf.Variable(wt_init([neurons_h1,n_outputs]),dtype = tf.float32, name = 'w2_fc')
    
    b1_fc = tf.Variable(tf.zeros((neurons_h1)),dtype= tf.float32,name = 'b1_fc')
    b2_fc = tf.Variable(tf.zeros((n_outputs)),dtype = tf.float32,name = 'b2_fc')
    
    # weights and bias for sparse_ae combination fully connected
    w1 = tf.Variable(wt_init([n_x,neurons_h1]), dtype = tf.float32,name = 'w1')
    w2 = tf.Variable(wt_init([neurons_h1,n_outputs]),dtype=tf.float32,name = 'w2')
    
    w_ae = tf.Variable(wt_init([neurons_h1,n_x]),dtype=tf.float32, name='w_ae')
    
    b1 = tf.Variable(tf.zeros(neurons_h1),dtype = tf.float32, name = 'b1')
    b2 = tf.Variable(tf.zeros(n_outputs),dtype=tf.float32,name = 'b2')
    
    b_ae = tf.Variable(tf.zeros(n_x),dtype=tf.float32,name = 'b_ae')
    
    parameters = {            
            'w1_fc':w1_fc,
            'w2_fc':w2_fc,
            'w1':w1,
            'w2':w2,
            'w_ae':w_ae,
            'b1_fc':b1_fc,
            'b2_fc':b2_fc,
            'b1':b1,
            'b2':b2,
            'b_ae':b_ae}
    return parameters

def fwd_propagation(x_ph,parameters):
    
      
    w1_fc = parameters['w1_fc']
    w2_fc = parameters['w2_fc'] 
    w1 = parameters['w1'] 
    w2 = parameters['w2']  
    w_ae = parameters['w_ae'] 
    b1_fc = parameters['b1_fc']  
    b2_fc = parameters['b2_fc']  
    b1 = parameters['b1']  
    b2 = parameters['b2'] 
    b_ae = parameters['b_ae'] 
    
    l1_fc = tf.add(tf.matmul(x_ph,w1_fc),b1_fc)
    l1_act_fc = tf.nn.relu(l1_fc)  
    logits_fc = tf.add(tf.matmul(l1_act_fc,w2_fc),b2_fc)
    
    l1 = tf.add(tf.matmul(x_ph,w1),b1)
    l1_act = tf.nn.sigmoid(l1)    
    logits = tf.add(tf.matmul(l1_act,w2),b2)
    
    x_hat = tf.nn.sigmoid(tf.add(tf.matmul(l1_act,w_ae),b_ae))
    
    return logits_fc,logits,x_hat,l1_act
  
def compute_cost_fc(logits_fc,y_ph,parameters,reg_term_lambda):
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_ph,logits=logits_fc)
    
    #w1_fc = parameters['w1_fc']
    #w2_fc = parameters['w2_fc']
    #loss+= reg_term_lambda*(tf.nn.l2_loss(w1_fc) + tf.nn.l2_loss(w2_fc))
    
    return tf.reduce_mean(loss)

def compute_cost_ae_fc(logits,y_ph,parameters,reg_term_lambda):
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_ph,logits=logits)
    
    #w1 = parameters['w1'] 
    #w2 = parameters['w2']
    #loss+= reg_term_lambda*(tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2))
    
    return tf.reduce_mean(loss)

def kl_divergence(p,p_hat):
    return p*tf.log(p)-p*tf.log(p_hat) + (1-p)*tf.log(1-p)-(1-p)*tf.log(1-p_hat)


def compute_cost_ae(x_hat,x_ph,parameters,reg_term_lambda,l1_act,rho,beta):
    
    diff = x_hat - x_ph   
    
    p_hat = tf.reduce_mean(tf.clip_by_value(l1_act,1e-10,1.0,name='clipper'),axis=0)
    kl = kl_divergence(rho,p_hat)
    
    w1 = parameters['w1']
    w_ae = parameters['w_ae'] 
    l2_loss = reg_term_lambda*(tf.nn.l2_loss(w1)+tf.nn.l2_loss(w_ae))
    
    loss = tf.reduce_mean(tf.reduce_sum(diff** 2,axis=1))+beta*tf.reduce_sum(kl)+l2_loss
    
    return loss
        
def model(tr_x,tr_y,te_x,te_y,learning_rate =1e-3,epochs = 1000,reg_term_lambda=1e-3,rho=0.1,beta=3):
    
    # tensorflow essentials
    tf.reset_default_graph()
    tf.set_random_seed(1)
    
    # getting the dimensions from the tr_x and tr_y
    m = tr_x.shape[0]
    n_x = tr_x.shape[1]
    n_y = np.max(tr_y)-np.min(tr_y) + 1
    
    # initialising cost list for different costs
    cost_fc_li = []
    cost_ae_li = []
    cost_ae_fc_li = []
    
    # creating placeholders for the graph
    x_ph,y_ph = create_placeholder(n_x)
    
    # parameter initialisation
    parameters = initialise_parameter(n_x,n_y)
    
    # getting logits of 3 different NN
    logits_fc,logits,x_hat,l1_act = fwd_propagation(x_ph,parameters) #l1_act used in cost_ae
    
    # cost calculations
    cost_fc = compute_cost_fc(logits_fc,y_ph,parameters,reg_term_lambda)
    cost_ae_fc = compute_cost_ae_fc(logits,y_ph,parameters,reg_term_lambda)
    cost_ae = compute_cost_ae(x_hat,x_ph,parameters,reg_term_lambda,l1_act,rho,beta)
    
    # optimizers
    optimizer_fc = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_fc)
    optimizer_ae_fc = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_ae_fc)
    optimizer_ae = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_ae)
    
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        
        print('Training sparse autoencoder ')      
        for epoch in range(epochs):
            
            _,c_ae = sess.run([optimizer_ae,cost_ae],feed_dict={x_ph:tr_x})
            cost_ae_li.append(c_ae)
            
            if (epoch+1)%200 == 0:
                print("Epoch: ",(epoch+1),"cost = ","{:.3f}".format(c_ae))
        
        print('Training softmax classifier with autoencoder feature representation.')
        for epoch in range(epochs):
            
            _,c_ae_fc = sess.run([optimizer_ae_fc,cost_ae_fc],feed_dict={x_ph:tr_x,y_ph:tr_y})
            cost_ae_fc_li.append(c_ae_fc)
            
            if (epoch+1)%200 == 0:
                print("Epoch: ",(epoch+1),"cost = ","{:.3f}".format(c_ae_fc))
        
        print('Training fully connected(fc) network of dimensions [784,200,10].')
        for epoch in range(epochs):
            
            _,c_fc = sess.run([optimizer_fc,cost_fc],feed_dict={x_ph:tr_x,y_ph:tr_y})
            cost_fc_li.append(c_fc)
            
            if (epoch+1)%200 == 0:
                print("Epoch: ",(epoch+1),"cost = ","{:.3f}".format(c_fc))
        
        correct_pred_ae_fc = tf.equal(tf.argmax(logits,axis=1),y_ph)
        acc_ae_fc = tf.reduce_mean(tf.cast(correct_pred_ae_fc,'float'))
        
        correct_pred_fc = tf.equal(tf.argmax(logits_fc,axis=1),y_ph)
        acc_fc = tf.reduce_mean(tf.cast(correct_pred_fc,'float'))
        
        print("Accuracy of fc NN on test set",acc_fc.eval({x_ph:te_x,y_ph:te_y}))
        print("Accuracy of ae feature fc NN on test set",acc_ae_fc.eval({x_ph:te_x,y_ph:te_y}) )
        

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
    
    #tr_y = tr_y.reshape(-1,1) # reshaping from (1000,) to (1000,1)
    #te_y = te_y.reshape(-1,1) # reshaping from (1000,) to (1000,1)
    
    model(tr_x,tr_y,te_x,te_y)
    
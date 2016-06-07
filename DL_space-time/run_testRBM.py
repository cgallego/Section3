# -*- coding: utf-8 -*-
"""
Created on Thu Jun 02 12:50:36 2016

@author: DeepLearning
"""

from DeepLearnFuncs import *

def run_RBM():
    # start by importing Deep Learning Funcs
    funcs = DeepLearnFuncs()
    
    learning_rate= 0.05 #0.25
    n_hidden=500
    batch_size=10 #20 
    training_epochs=50 #50 
    
    n_chains=20
    n_samples=10
    output_folder='rbm_plots_'+str(learning_rate)+'_'+str(n_hidden)+'_'+str(batch_size)
    
             
    ############
    # train Stacked dAutoencoder                 
    ############
    funcs.test_rbm(learning_rate, training_epochs,  batch_size,
             n_chains, n_samples, output_folder, n_hidden)
     
    

   
if __name__ == '__main__':
    run_RBM()

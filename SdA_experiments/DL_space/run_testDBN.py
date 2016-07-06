# -*- coding: utf-8 -*-
"""
Created on Thu Jun 02 12:50:36 2016

@author: DeepLearning
"""

from DeepLearnFuncs import *

def run_DBN():
    # start by importing Deep Learning Funcs
    funcs = DeepLearnFuncs()
    
    finetune_lr = 0.10
    pretraining_epochs = 50
    pretrain_lr = 0.75
    
    k = 1 # provide k (the number of Gibbs steps to perform in CD or PCD)
    training_epochs = 1000
    batch_size=1
    nhidden = 1600
    filtsize = 20 # 4*12*12 = 576
    plots_folder = 'dbn_plots_hn'+str(nhidden)
    
    ############
    # train Deep Belief Networks
    ############    
    funcs.test_DBN(nhidden, filtsize, finetune_lr, pretraining_epochs,
             pretrain_lr, k, training_epochs, batch_size, plots_folder)
   
if __name__ == '__main__':
    run_DBN()

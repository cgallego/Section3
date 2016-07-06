# -*- coding: utf-8 -*-
"""
Created on Wed Jun 01 15:27:03 2016

@author: DeepLearning
"""

from DeepLearnFuncs import *

def run_dAutoencoder():
    # start by importing Deep Learning Funcs
    funcs = DeepLearnFuncs()
    
    learning_rate=0.0095
    training_epochs=200
    corruptionL=0.0
    batch_size=2
    output_folder='plots/dA_plots'
    
    ############
    # train dAutoencoder                 
    ############
    dfLLdata = funcs.test_dA(learning_rate, training_epochs, corruptionL, batch_size, output_folder)
     
    ############
    ### plotting or cost
    ### the cost we minimize during training is the negative log likelihood of
    ############
    x = dfLLdata['iter'].values
    y = dfLLdata['LL_iter'].values
    plt.figure()
    plt.plot(x, y, 'bo--')
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('dA Cost', fontsize=14)
    plt.title('MLP: learning_rate = '+str(learning_rate)+' batch_size = '+str(batch_size), fontsize=14)

          
    # change correpution level to 0.25
    corruptionL=0.25
    
    ############
    # train dAutoencoder                 
    ############
    dfLLdata = funcs.test_dA(learning_rate, training_epochs, corruptionL, batch_size, output_folder)
    
    ############
    ### plotting or cost
    ### the cost we minimize during training is the negative log likelihood of
    ############
    x = dfLLdata['iter'].values
    y = dfLLdata['LL_iter'].values
    plt.figure()
    plt.plot(x, y, 'bo--')
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('dA Cost', fontsize=14)
    plt.title('MLP: learning_rate = '+str(learning_rate)+' batch_size = '+str(batch_size), fontsize=14)

    # change correpution level to 0.25
    corruptionL=0.6
    
    ############
    # train dAutoencoder                 
    ############
    dfLLdata = funcs.test_dA(learning_rate, training_epochs, corruptionL, batch_size, output_folder)
    
    ############
    ### plotting or cost
    ### the cost we minimize during training is the negative log likelihood of
    ############
    x = dfLLdata['iter'].values
    y = dfLLdata['LL_iter'].values
    plt.figure()
    plt.plot(x, y, 'bo--')
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('dA Cost', fontsize=14)
    plt.title('MLP: learning_rate = '+str(learning_rate)+' batch_size = '+str(batch_size), fontsize=14)



if __name__ == '__main__':
    run_dAutoencoder()

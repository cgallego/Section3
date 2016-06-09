# -*- coding: utf-8 -*-
"""
Created on Thu Jun 02 12:50:36 2016

@author: DeepLearning
"""

from DLFuncs_SdA import *
    
def run_testSdA_timep():
    # start by importing Deep Learning Funcs
    funcs = DLFuncs_SdA()
    
    pretraining_epochs = 100
    pretrain_lr = 0.45
    batch_size = 500
    
    training_epochs = 1000
    finetune_lr = 0.075    
    
    output_folder = 'SdA_plots_subs'
    corruption_levels=[0.30, 0.4, 0.50, 0.60]
    hidden_layers_sizes=[2025,1600,1225,900]
    hidden_layers_sidelen = [30,45,40,35,30]
    
    ############
    # train Stacked dAutoencoder                 
    ############
    dfpredata, dfinedata = funcs.test_SdA_timep(pretraining_epochs, pretrain_lr, batch_size,
                                                training_epochs, finetune_lr, 
                                                corruption_levels, 
                                                hidden_layers_sizes, hidden_layers_sidelen, output_folder)
    ############
    ### plotting or cost
    ### the cost we minimize during training is the negative log likelihood of
    ############
    plt.figure()
    sns.lmplot('iter', 'LL_iter', data=dfpredata, hue='layer', fit_reg=False)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('dA Cost', fontsize=14)
    plt.title('Pretraining Stacked dAutoencoder learn_rate = '+str(pretrain_lr)+' pretrain_epochs = '+str(pretraining_epochs), fontsize=14)

    ############
    ### plotting likelihood or cost FIne tunning
    ### the cost we minimize during training is the negative log likelihood of
    ############
    x = dfinedata['iter'].values
    y = dfinedata['LL_iter'].values
    plt.figure()
    plt.plot(x, y, 'bo--')
    plt.xlabel('iterations', fontsize=14)
    plt.ylabel('negative log likelihood', fontsize=14)
    plt.title('Fine Tunning: finetune_lr = '+str(finetune_lr)+' batch_size = '+str(batch_size), fontsize=14)

    x = dfinedata['iter'].values
    y = dfinedata['loss'].values
    plt.figure()
    plt.plot(x, y, 'bo--')
    plt.xlabel('iterations')
    plt.ylabel('Validation 0-1-loss %')
    plt.title('Fine Tunning: finetune_lr = '+str(finetune_lr)+' batch_size = '+str(batch_size), fontsize=14)

   
if __name__ == '__main__':
    run_testSdA_timep()

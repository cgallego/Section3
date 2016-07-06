# -*- coding: utf-8 -*-
"""
Created on Tue Jul 05 09:07:38 2016

@author: DeepLearning
"""

import six.moves.cPickle as pickle
from DLFuncs_2D_temporal import *

import seaborn as sns
import pandas as pd
import numpy as np

import itertools
    
def run_testSdA_timep():
    # start by importing Deep Learning Funcs
    funcs = DLFuncs_2D_temporal()
    
    pretraining_epochs = 100
    training_epochs = 100
    finetune_lr = 0.9   
    output_folder = '2D_temporal_gridSearch_results'
    
    ############
    ### Define grid search parameters
    ############
    pretrain_lr = [0.01,0.1]
    noise_levels =[0.35,0.5] # [0.10, 0.25, 0.50]
    nlayers = [1,3]
    hidden_layers_sizes = [225,625,1225]  # [225,324,900]
    hidden_layers_sidelen = [15,25,35] # [30,15,18,30]
    batch_sizes = [250,50,10]
    k=0
    BestAveAccuracy = 81
    
    ###########
    ## Process Resuts
    ###########
    #dfresults = pd.DataFrame() # when first time
    pkl_filegridS = open(output_folder+'/2D_temporal_gridSearch_results.pkl','rb')
    dfresults = pickle.load(pkl_filegridS)
    print dfresults
    #items = itertools.product(nlayers, hidden_layers_sizes, noise_levels, pretrain_lr, batch_sizes)
    #item=items.next()
    
    for item in itertools.product(nlayers, hidden_layers_sizes, noise_levels, pretrain_lr, batch_sizes): 
        k+=1
        print(k,item)
        
        if(k>9): #k-1 195
            #break
            # setup the training functions
            nlayer = item[0]
            nhidden = item[1]
            sidelen = hidden_layers_sidelen[ hidden_layers_sizes.index(nhidden) ]
            noiserate = item[2]
            pretrain_lr = item[3]
            finetune_lr = pretrain_lr
            batch_size = item[4]
            if(nlayer == 1):
                StackedDA_layers = [nhidden]
                corruption_levels = [noiserate]
                hidden_sidelen = [sidelen]
            if(nlayer == 2):
                StackedDA_layers = [nhidden,nhidden]
                corruption_levels = [noiserate,noiserate]
                hidden_sidelen = [sidelen,sidelen]
            if(nlayer == 3):
                StackedDA_layers = [nhidden,nhidden,nhidden]
                corruption_levels = [noiserate,noiserate,noiserate]
                hidden_sidelen = [sidelen,sidelen,sidelen]
            if(nlayer == 4):
                StackedDA_layers = [nhidden,nhidden,nhidden,nhidden]
                corruption_levels = [noiserate,noiserate,noiserate,noiserate]
                hidden_sidelen = [sidelen,sidelen,sidelen,sidelen]
                    
            ############
            # train Stacked dAutoencoder                 
            ############
            dfpredata, dfinedata, imagefilters, sda, Acutrain0, Acutrain1, Acuvalid0, Acuvalid1, Acutest0, Acutest1 = funcs.test_SdA_timep(pretraining_epochs, 
                                                        pretrain_lr, batch_size,
                                                        training_epochs, finetune_lr, 
                                                        corruption_levels, 
                                                        StackedDA_layers, hidden_sidelen, output_folder)                                            
            ############                                         
            # prepare display    
            ############                                                                                                 
            fig, axes = plt.subplots(ncols=2, nrows=2)
            axes[0,0].imshow(imagefilters[0], cmap="Greys_r")
            axes[0,1].imshow(imagefilters[1], cmap="Greys_r")
            axes[1,0].imshow(imagefilters[2], cmap="Greys_r")
            axes[1,1].imshow(imagefilters[3], cmap="Greys_r")  
            axes[0,0].get_xaxis().set_visible(False)
            axes[0,0].get_yaxis().set_visible(False)
            axes[0,1].get_xaxis().set_visible(False)
            axes[0,1].get_yaxis().set_visible(False)
            axes[1,0].get_xaxis().set_visible(False)
            axes[1,0].get_yaxis().set_visible(False)
            axes[1,1].get_xaxis().set_visible(False)
            axes[1,1].get_yaxis().set_visible(False)
            plt.show() 
        
            #show and save                     
            fig.savefig(output_folder+'/2Dtempfilters_layers_'+str(item)+'.pdf')
                
                                   
            ############
            ### plotting or cost
            ### the cost we minimize during training is the negative log likelihood of
            ###########
            plt.figure()
            sns.lmplot('iter', 'LL_iter', data=dfpredata, hue='layer', fit_reg=False)
            plt.xlabel('epoch', fontsize=14)
            plt.ylabel('dA Cost', fontsize=14)
            plt.title('Pretraining Stacked dAutoencoder learn_rate = '+str(pretrain_lr)+' pretrain_epochs = '+str(pretraining_epochs), fontsize=14)
            plt.savefig(output_folder+'/Pretraining_SdA_'+str(item)+'.pdf')
            
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
            plt.savefig(output_folder+'/FineTunningSdA_'+str(item)+'.pdf')
        
            x = dfinedata['iter'].values
            y = dfinedata['loss'].values
            plt.figure()
            plt.plot(x, y, 'bo--')
            plt.xlabel('iterations')
            plt.ylabel('Validation 0-1-loss %')
            plt.title('Fine Tunning: finetune_lr = '+str(finetune_lr)+' batch_size = '+str(batch_size), fontsize=14)
            plt.savefig(output_folder+'/FineTunningSdA_loss_'+str(item)+'.pdf')
    
            # append results
            itemresults = item + (Acutrain0, Acutrain1, Acuvalid0, Acuvalid1, Acutest0, Acutest1)
            dSresultsiter =  pd.DataFrame(data=np.array(itemresults)).T
            dSresultsiter.columns=['nlayers', 'nhiddens', 'nnoise_rate', 'lr_pretrain', 'batch_size', 'Acutrain0', 'Acutrain1', 'Acuvalid0', 'Acuvalid1', 'Acutest0', 'Acutest1']
              
            dfresults = dfresults.append(dSresultsiter)      
            AveAccuracy = (Acuvalid0 + Acuvalid1)/2
            # find best model so far
            if( AveAccuracy > BestAveAccuracy):
                bestsDA = sda
                BestAveAccuracy = AveAccuracy
                print("best Accuracy = %d, for SdA:" % BestAveAccuracy)
                print(bestsDA)
                
                #####################################
                # save the best model
                #####################################
                with open(output_folder+'/bestsDA.obj', 'wb') as fp:
                    pickle.dump(bestsDA, fp)
            
            # save the best model
            with open(output_folder+'/2D_temporal_gridSearch_results.pkl', 'wb') as f:
                pickle.dump(dfresults, f)

    ### continue
    print(dfresults)
        
        

   
if __name__ == '__main__':
    run_testSdA_timep()
    
    #####################################
    # Open best model
    #####################################
    with open(output_folder+'/bestsDA.obj', 'rb') as fp:
        bestsDA = pickle.load(fp)
        
    ###########
    ## Process Resuts
    ###########
    import six.moves.cPickle as pickle
    output_folder = '2D_temporal_gridSearch_results'
    pkl_filegridS = open(output_folder+'/2D_temporal_gridSearch_results.pkl','rb')
    dfresults = pickle.load(pkl_filegridS)
    print dfresults

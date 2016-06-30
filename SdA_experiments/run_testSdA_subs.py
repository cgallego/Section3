# -*- coding: utf-8 -*-
"""
Created on Thu Jun 02 12:50:36 2016

@author: DeepLearning
"""
import six.moves.cPickle as pickle
from DLFuncs_SdA import *

import seaborn as sns
import pandas as pd
import numpy as np

import itertools
    
def run_testSdA_timep():
    # start by importing Deep Learning Funcs
    funcs = DLFuncs_SdA()
    
    pretraining_epochs = 500
    training_epochs = 100
    finetune_lr = 0.25    
    output_folder = 'finaltheanoSdA' # 'SdA_plots_subs_3layers'
    
    ############
    ### Define grid search parameters
    ############
    pretrain_lr = [0.0001,0.001,0.01,0.1]
    noise_levels =[0.20,0.40] # [0.10, 0.25, 0.50]
    nlayers = [1,2,3]
    hidden_layers_sizes = [225,400,900,1600]  # [225,324,900]
    hidden_layers_sidelen = [15,20,30,40] # [30,15,18,30]
    batch_sizes = [1000,500,100,10]
    
    k=0
    BestAveAccuracy = 0
    
    ###########
    ## Process Resuts
    ###########
    #dfresults = pd.DataFrame() # when first time
    pkl_filegridS = open(output_folder+'/gridSearch_results.pkl','rb')
    dfresults = pickle.load(pkl_filegridS)
    print dfresults
    #items = itertools.product(nlayers, hidden_layers_sizes, noise_levels, pretrain_lr, batch_sizes)
    #item=items.next()
    
    for item in itertools.product(nlayers, hidden_layers_sizes, noise_levels, pretrain_lr, batch_sizes): 
        k+=1
        print(k,item)
        
        if(k>55): #k-1
            # setup the training functions
            nlayer = item[0]
            nhidden = item[1]
            sidelen = hidden_layers_sidelen[ hidden_layers_sizes.index(nhidden) ]
            noiserate = item[2]
            pretrain_lr = item[3]
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
                    
            ############
            # train Stacked dAutoencoder                 
            ############
            dfpredata, dfinedata, sda, Acutrain0, Acutrain1, Acuvalid0, Acuvalid1, Acutest0, Acutest1 = funcs.test_SdA_timep(pretraining_epochs, 
                                                        pretrain_lr, batch_size,
                                                        training_epochs, finetune_lr, 
                                                        corruption_levels, 
                                                        StackedDA_layers, hidden_sidelen, output_folder)
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
                print("best Accuracy = %d, for SdA:" % AveAccuracy)
                print(bestsDA)
                
                #####################################
                # save the best model
                #####################################
                with open('finaltheanoSdA/bestsDA.obj', 'wb') as fp:
                    pickle.dump(bestsDA, fp)
            
            # save the best model
            with open('finaltheanoSdA/gridSearch_results.pkl', 'wb') as f:
                pickle.dump(dfresults, f)
            
    ### continue
    print(dfresults)
        
        

   
if __name__ == '__main__':
    run_testSdA_timep()
    
    #####################################
    # Open best model
    #####################################
    with open('finaltheanoSdA/bestsDA.obj', 'rb') as fp:
        bestsDA = pickle.load(fp)
        
    ###########
    ## Process Resuts
    ###########
    import six.moves.cPickle as pickle
    pkl_filegridS = open('finaltheanoSdA/gridSearch_results.pkl','rb')
    dfresults = pickle.load(pkl_filegridS)
    print dfresults

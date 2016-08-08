# -*- coding: utf-8 -*-
"""
Created on Tue Jul 05 09:07:38 2016

@author: DeepLearning
"""

import six.moves.cPickle as pickle
from DLRBM_2D_temporal import *

import seaborn as sns
import pandas as pd
import numpy as np

import itertools
    
def run_testSdA_timep():
    # start by importing Deep Learning Funcs
    funcs = DLRBM_2D_temporal()
    
    pretraining_epochs = 100
    training_epochs = 100
    finetune_lr = 0.1   
    output_folder = 'rbm_plots'
    
    ############
    ### Define grid search parameters
    ############
    pretrain_lr = [0.01,0.1]
    noise_levels =[0.35,0.5] # [0.10, 0.25, 0.50]
    nlayers = [1,2,3]
    hidden_layers_sizes = [400,625,900]  # [225,625,400] + 100
    hidden_layers_sidelen = [20,25,30] # [30,15,18,30]
    batch_sizes = [50,10,2]
    k=0
    BestAveAccuracy = 0 ###83.199
    
    n_chains=20
    n_samples=10    
    n_hidden=500
    output_folder='rbm_plots'
    learning_rate=0.1
    training_epochs=15

    ###########
    ## Process Resuts
    ###########
    dfresults = pd.DataFrame() # when first time
#    pkl_filegridS = open(output_folder+'/2D_temporal_gridSearch_results_subs_smaller.pkl','rb')
#    dfresults = pickle.load(pkl_filegridS)
#    print dfresults
    #items = itertools.product(nlayers, hidden_layers_sizes, noise_levels, pretrain_lr, batch_sizes)
    #item=items.next()
    
    for item in itertools.product(nlayers, hidden_layers_sizes, noise_levels, pretrain_lr, batch_sizes): 
        k+=1
        print(k,item)
        
        if(k>0): #k-1 195
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
            dfpredata, dfinedata, imagefilters, sda, Acutrain0, Acutrain1, Acuvalid0, Acuvalid1, Acutest0, Acutest1 = funcs.test_RBM_timep(pretraining_epochs, 
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
            AveAccuracy = (Acuvalid0 + Acuvalid1)
            AveAccuracy = AveAccuracy/2
            # find best model so far
            if( AveAccuracy > BestAveAccuracy):
                bestsDA = sda
                BestAveAccuracy = AveAccuracy
                print("best Accuracy = %d, for SdA:" % BestAveAccuracy)
                print(bestsDA)
                
                #####################################
                # save the best model
                #####################################
                with open(output_folder+'/bestsDA_subs_smaller.obj', 'wb') as fp:
                    pickle.dump(bestsDA, fp)
            
            # save the best model
            with open(output_folder+'/2D_temporal_gridSearch_results_subs_smaller.pkl', 'wb') as f:
                pickle.dump(dfresults, f)

    ### continue
    print(dfresults)
        
        

   
if __name__ == '__main__':
    run_testSdA_timep()
        
    import six.moves.cPickle as pickle
    import numpy as np
    output_folder = '2D_temporal_gridSearch_results_subs_smaller'
    
    #####################################
    # Open best model
    #####################################
    with open(output_folder+'/bestsDA_subs_smaller.obj', 'rb') as fp:
        bestsDA = pickle.load(fp)
        
    ###########
    ## Process Resuts
    ###########    
    pkl_filegridS = open(output_folder+'/2D_temporal_gridSearch_results_subs_smaller.pkl','rb')
    dfresults = pickle.load(pkl_filegridS)
    print dfresults
    
    # find best perfoming params
    avergAccu = (dfresults['Acuvalid0'])+np.asarray(dfresults['Acuvalid1'])
    avergAccu = avergAccu/2
    bestAccu = dfresults[avergAccu==np.max(avergAccu)]
    print "best performing on Acuvalid parameters SdA"
    print bestAccu, max(avergAccu)

    
    # find best perfoming params
    avergAccu = (dfresults['Acutest0'])+np.asarray(dfresults['Acutest1'])
    avergAccu = avergAccu/2
    bestAccu = dfresults[avergAccu==np.max(avergAccu)]
    print "best performing on Acutest parameters SdA"
    print bestAccu, max(avergAccu)

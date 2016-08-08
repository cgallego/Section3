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
 
    
def run_testDBN_timep():
    # start by importing Deep Learning Funcs
    funcs = DLRBM_2D_temporal()
      
    ############
    ### Define grid search parameters
    ############
    output_folder='DBN_plots'
    pretraining_epochs=200
    training_epochs=500
    pretrain_lr = [0.1]
    finetune_lr= [0.1]
    kCD = [1] 
    nlayers = [1,2,3]
    hidden_layers_sizesList = [100,225,400,625]
    hidden_layers_sidelen = [10,15,20,25] # [30,15,18,30]
    batch_sizes = [50,10]
    k=0
    BestAveAccuracy = 63.9 ###83.199
    
    ###########
    ## Process Resuts
    ###########
    #dfresults = pd.DataFrame() # when first time
    pkl_filegridS = open(output_folder+'/2D_DBN_gridSearch_results_10x10x5.pkl','rb')
    dfresults = pickle.load(pkl_filegridS)
    print dfresults
    #items = itertools.product(nlayers, hidden_layers_sizes, noise_levels, pretrain_lr, batch_sizes)
    #item=items.next()
    
    for item in itertools.product(nlayers, hidden_layers_sizesList, pretrain_lr, finetune_lr, kCD, batch_sizes): 
        k+=1
        print(k,item)
        
        if(k>3): #k-1 195
            #break
            # setup the training functions
            nlayer = item[0]
            nhidden = item[1]
            sidelen = hidden_layers_sidelen[ hidden_layers_sizesList.index(nhidden) ]
            pretrain_lr = item[2]
            finetune_lr = item[3]
            kCD = item[4]
            batch_size = item[5]
            
            if(nlayer == 1):
                hidden_layers_sizes = [nhidden]
            if(nlayer == 2):
                hidden_layers_sizes = [nhidden,nhidden]
            if(nlayer == 3):
                hidden_layers_sizes = [nhidden,nhidden,nhidden]
            if(nlayer == 4):
                hidden_layers_sizes = [nhidden,nhidden,nhidden,nhidden]
                    
            ############
            # train Stacked dAutoencoder                 
            ############
            dfpredata, dfinedata, dbn, Acutrain0, Acutrain1, Acuvalid0, Acuvalid1, Acutest0, Acutest1 = funcs.test_DBN(finetune_lr=finetune_lr, 
                                                            pretraining_epochs=pretraining_epochs,
                                                            pretrain_lr=pretrain_lr, k=kCD, training_epochs=training_epochs, batch_size=batch_size,
                                                            hidden_layers_sizes=hidden_layers_sizes, sidelen=sidelen,
                                                            output_folder=output_folder, item=item)                                
                                   
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
            dSresultsiter.columns=['nlayers', 'nhiddens', 'lr_pretrain', 'finetune_lr', 'kCD', 'batch_size', 'Acutrain0', 'Acutrain1', 'Acuvalid0', 'Acuvalid1', 'Acutest0', 'Acutest1']
              
            dfresults = dfresults.append(dSresultsiter)      
            AveAccuracy = (Acuvalid0 + Acuvalid1)
            AveAccuracy = AveAccuracy/2
            # find best model so far
            if( AveAccuracy > BestAveAccuracy):
                bestDBN = dbn
                BestAveAccuracy = AveAccuracy
                print("best Accuracy = %d, for DBN:" % BestAveAccuracy)
                print(bestDBN)
                
                #####################################
                # save the best model
                #####################################
                with open(output_folder+'/bestDBN_10x10x5.obj', 'wb') as fp:
                    pickle.dump(bestDBN, fp)
            
            # save the best model
            with open(output_folder+'/2D_DBN_gridSearch_results_10x10x5.pkl', 'wb') as f:
                pickle.dump(dfresults, f)

    ### continue
    print(dfresults)
        
        

   
if __name__ == '__main__':
    run_testDBN_timep()
        
    import six.moves.cPickle as pickle
    import numpy as np
    output_folder = 'DBN_plots'
    
    #####################################
    # Open best model
    #####################################
    with open(output_folder+'/bestDBN_10x10x5.obj', 'rb') as fp:
        bestDBN = pickle.load(fp)
        
    ###########
    ## Process Resuts
    ###########    
    pkl_filegridS = open(output_folder+'/2D_DBN_gridSearch_results_10x10x5.pkl','rb')
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

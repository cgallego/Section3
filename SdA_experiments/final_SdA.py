
from DLFuncs_SdA import *
import utils
import utils2
import seaborn as sns
import pandas as pd
import numpy as np

import six.moves.cPickle as pickle

import Noise
import Layers
import matplotlib.pyplot as plt
import random


class Trainer():
    def train(self, layers, X, y, epochs, batchsize):
        #rows = np.arange(X.shape[0])
        rows_batch = random.sample(xrange(X.shape[0]), batchsize)
        dA_avg_costs = []
        dA_costs = []
        
        ## stochastic sampler in n_iterations
        for iters in range(epochs):
            numpy.random.shuffle(rows_batch)
            for idx in rows_batch:
                tmp = X[[idx]]
                for i in range(len(layers)):
                    tmp = layers[i].activate(tmp)
                    
                grad, dA_avg_costs = layers[-1].backwardPass(y[[idx]], dA_avg_costs)
            
                for i in range(len(layers)-1, -1, -1):
                    grad = layers[i].update(grad)
                    
            print('epoch %i, error %f' % (iters, np.mean(dA_avg_costs) ))    
            dA_costs.append( np.mean(dA_avg_costs) )
            
        return layers, dA_costs
        
        
class StackedDA():
    def __init__(self, structure, alpha=0.01):
        self.alpha = alpha
        self.structure = structure
        self.Layers = []
        print "Call \"pre_train(epochs)\""
        print "\nStacked Denoising Autoencoder Structure:\t%s"%(" -> ".join([str(x) for x in self.structure]))
        
    def __getstate__(self):
        print "I'm being pickled"
        return self.__dict__
    
    def __setstate__(self, d):
        print "I'm being unpickled with these values:", d
        self.__dict__ = d
        
    def pre_train(self, X, epochs=1, batchsize=100, noise_rate=0.3):
        self.structure = numpy.concatenate([[X.shape[1]], self.structure])
        self.X = X
        trainer = Trainer()
        print "Pre-training: "#, self.__repr__()
        for i in range(len(self.structure) - 1):
            print "Layer: %dx%d"%( self.structure[i], self.structure[i+1])
            s1 = Layers.SigmoidLayer(self.structure[i], self.structure[i+1], noise=Noise.GaussianNoise(noise_rate))
            s2 = Layers.SigmoidLayer(self.structure[i+1], self.X.shape[1])

            #########################
            # PRETRAINING THE MODEL #
            #########################
            layers, dA_costs = trainer.train([s1, s2], self.X, self.X, epochs, batchsize)
            s1, s2 = layers
            self.X = s1.activate(self.X)
            self.Layers.append(s1)
            
            ##############
            # Format      
            #################           
            LLdata = [float(L) for L in dA_costs]
            LLiter = [float(it) for it in range(epochs)]
            dfpredata = pd.DataFrame( LLdata )
            dfpredata.columns = ['dA_avg_costs']
            dfpredata['iter'] = LLiter
            
            ############
            ### plotting or cost
            ### the cost we minimize during training is the negative log likelihood of
            ############
            plt.figure()
            sns.lmplot('iter', 'dA_avg_costs', data=dfpredata, fit_reg=False)
            plt.xlabel('epoch', fontsize=14)
            plt.ylabel('dA error', fontsize=14)
            plt.title('SdA_structure_'+str([self.structure])+'_noise_'+str(noise_rate)+'_pretrain_epochs_'+str(epochs), fontsize=9)
            plt.savefig('finalSdA/'+'SdA_structure_'+str([self.structure])+'_noise_'+str(noise_rate)+'_pretrain_epochs_'+str(epochs)+'.png')

    
    def finalLayer(self, X, y, epochs=1):
        print "Final Layer" 
        V = self.predict(X)
        softmax = Layers.SoftmaxLayer(self.Layers[-1].W.shape[1], y.shape[1]) 
        
        #########################
        # Final layer of THE MODEL #
        #########################
        batchsize = X.shape[0]
        softmax, dA_avg_perm = Trainer().train([softmax], V, y, epochs, batchsize)
        self.Layers.append(softmax[0])
        
        ##############
        # Format      
        #################           
        LLdata = [float(L) for L in dA_avg_perm]
        LLiter = [float(it) for it in range(epochs)]
        dfpredata = pd.DataFrame( LLdata )
        dfpredata.columns = ['dA_avg_costs']
        dfpredata['iter'] = LLiter
        
        ############
        ### plotting or cost
        ### the cost we minimize during training is the negative log likelihood of
        ############
        plt.figure()
        sns.lmplot('iter', 'dA_avg_costs', data=dfpredata, fit_reg=False)
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('softmax error', fontsize=14)
        plt.title('softmax_train_epochs_'+str(epochs), fontsize=9)
        plt.savefig('finalSdA/'+'softmax_train_epochs_'+str(epochs)+'.png')

     
    def fine_tune(self, X, y, epochs=1):
        print "Fine Tunning" 
        #########################
        # Fine Tunning THE MODEL #
        #########################
        batchsize = X.shape[0]
        self.Layers, dA_avg_perm = Trainer().train(self.Layers, X, y, epochs, batchsize)
        
        ##############
        # Format      
        #################           
        LLdata = [float(L) for L in dA_avg_perm ]
        LLiter = [float(it) for it in range(epochs)]
        dfinedata = pd.DataFrame( LLdata )
        dfinedata.columns = ['dA_avg_costs']
        dfinedata['iter'] = LLiter
        
        ############
        ### plotting or cost
        ### the cost we minimize during training is the negative log likelihood of
        ############
        plt.figure()
        sns.lmplot('iter', 'dA_avg_costs', data=dfinedata, fit_reg=False)
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('finetune error', fontsize=14)
        plt.title('fine_tune_structure_'+str([self.structure])+'_train_epochs_'+str(epochs), fontsize=9)
        plt.savefig('grid_searchResults/'+'fine_tune_structure_'+str([self.structure])+'_train_epochs_'+str(epochs)+'.png')

     
    def predict(self, X):
        #print self self=sDA
        tmp = X
        for L in self.Layers:
            tmp = L.activate(tmp)
        return tmp
               
               
               

        
    
    
    
    
if __name__ == '__main__':
    funcs = DLFuncs_SdA()
    traindata_path='Z://Cristina//Section3//SdA_experiments//allLpatches.pklz'
    trainUdata_path='Z://Cristina//Section3//SdA_experiments//allUpatches.pklz'
    labeldata_path='Z://Cristina//Section3//SdA_experiments//allLabels.pklz'
    
    datasets = funcs.load_wUdata(traindata_path, labeldata_path, trainUdata_path)
 
    ############
    ### plotting labels 
    ############
    # get training data in numpy format   
    X,y = datasets[3]
    Xtrain = np.asarray(X)
    # extract one img
    Xtrain = Xtrain.reshape(Xtrain.shape[0], 4, 900)
    Xtrain = Xtrain[:,0,:]
    ytrain = utils2.makeMultiClass(y)
    
    # get valid data in numpy format   
    X,y = datasets[4]
    Xvalid = np.asarray(X)
    # extract one img
    Xvalid = Xvalid.reshape(Xvalid.shape[0], 4, 900)
    Xvalid = Xvalid[:,0,:]
    yvalid = utils2.makeMultiClass(y)
    
    # get training data in numpy format   
    X,y = datasets[5]
    Xtest = np.asarray(X)
    # extract one img
    Xtest = Xtest.reshape(Xtest.shape[0], 4, 900)
    Xtest = Xtest[:,0,:]
    ytest = utils2.makeMultiClass(y)
    
    ###########
    ## Process Resuts
    ###########
    pkl_filegridS = open('grid_searchResults/gridSearch_results.pkl','rb')
    dfresults = pickle.load(pkl_filegridS)
    print dfresults
    
    aver_perf = (np.array(dfresults['accuracy0'])+np.array(dfresults['accuracy1']))/2
    best_aver_perf = aver_perf[aver_perf == np.max(aver_perf)]
    print("best average Accuracy = %f, for SdA" % best_aver_perf)
    optimal_params = dfresults.iloc[list(aver_perf == np.max(aver_perf))]
    print optimal_params
    
    nlayer = optimal_params['nlayers'][0]
    nhidden = optimal_params['nhiddens'][0]
    noiseRate = optimal_params['nnoise_rate'][0]
    alpha =  optimal_params['nalpha'][0]
    
    ## Set up
    if(nlayer == 1):
        StackedDA_layers = [nhidden]
    if(nlayer == 2):
        StackedDA_layers = [nhidden,nhidden]
    if(nlayer == 3):
        StackedDA_layers = [nhidden,nhidden,nhidden]
        
    # building the SDA
    sDA = StackedDA(StackedDA_layers, alpha)
    
    # pre-trainning the SDA
    epochs=10
    batch_size=100
    sDA.pre_train(Xtrain, noise_rate=noiseRate, epochs=epochs, batchsize=batch_size)
    
    #####################################
    # saving a PNG representation of the first layer
    #####################################
    # Plot images in 2D       
    nhiddens = [400,625,900,1600,225]
    hidden_layers_sidelen = [20,25,30,40,15]
    sidelen = hidden_layers_sidelen[ nhiddens.index(nhidden) ]
    
    W0 = sDA.Layers[0].W[1:,:]
    imageW0 = Image.fromarray(
        utils.tile_raster_images(X=W0 , img_shape=(sidelen, sidelen), 
                           tile_shape=(10, 10),
                           tile_spacing=(1, 1)))

    #show and save                     
    imageW0.save('finalSdA/filters_1stlayer_'+str(StackedDA_layers)+'_'+str(float(noiseRate))+'_epochs_'+str(epochs)+'.png')
    # prepare display    
    fig, ax = plt.subplots()  
    ax.imshow(imageW0,  cmap="Greys_r")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    
    
    #####################################
    # adding the final layer
    #####################################
    sDA.finalLayer(Xtrain, ytrain, epochs=10)
        
    # trainning the whole network
    sDA.fine_tune(Xtrain, ytrain, epochs=10)

    #####################################
    # predicting using the SDA
    ##################################### 
    # let's see how the network did  
    pred = sDA.predict(Xvalid).argmax(1)
    y = yvalid.argmax(1)
    e0 = 0.0; y0 = len([0 for yi in range(len(y)) if y[yi]==0])
    e1 = 0.0; y1 = len([1 for yi in range(len(y)) if y[yi]==1])
    for i in range(len(y)):
        if(y[i] == 1):
            #print(y[i]==pred[i], y[i])
            e1 += y[i]==pred[i]
        if(y[i] == 0):
            #print(y[i]==pred[i], y[i])
            e0 += y[i]==pred[i]

    # printing the result, this structure should result in 80% accuracy
    print "valid accuracy for class 0: %2.2f%%"%(100*e0/y0)
    print "valid accuracy for class 1: %2.2f%%"%(100*e1/y1)
    
    pred = sDA.predict(Xtest).argmax(1)
    y = ytest.argmax(1)
    e0 = 0.0; y0 = len([0 for yi in range(len(y)) if y[yi]==0])
    e1 = 0.0; y1 = len([1 for yi in range(len(y)) if y[yi]==1])
    for i in range(len(y)):
        if(y[i] == 1):
            #print(y[i]==pred[i], y[i])
            e1 += y[i]==pred[i]
        if(y[i] == 0):
            #print(y[i]==pred[i], y[i])
            e0 += y[i]==pred[i]

    # printing the result, this structure should result in 80% accuracy
    print "test accuracy for class 0: %2.2f%%"%(100*e0/y0)
    print "test accuracy for class 1: %2.2f%%"%(100*e1/y1)
    
    
    #####################################
    # save the best model
    #####################################
    with open('finalSdA/sDA.obj', 'wb') as fp:
        pickle.dump(sDA, fp)
        
        
    #####################################
    # Open best model
    #####################################
    with open('finalSdA/sDA.obj', 'rb') as fp:
        sDA = pickle.load(fp)
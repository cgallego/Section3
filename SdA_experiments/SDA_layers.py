
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
            np.random.shuffle(rows_batch)
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
    def __init__(self, structure, alpha=0.01, item=''):
        self.alpha = alpha
        self.structure = structure
        self.Layers = []
        self.item = item
        print "Call \"pre_train(epochs)\""
        print "\nStacked Denoising Autoencoder Structure:\t%s"%(" -> ".join([str(x) for x in self.structure]))
        
    def __getstate__(self):
        print "I'm being pickled"
        return self.__dict__
    
    def __setstate__(self, d):
        print "I'm being unpickled with these values:", d
        self.__dict__ = d
        
    def pre_train(self, X, epochs=1, batchsize=100, noise_rate=0.3):
        self.structure = np.concatenate([[X.shape[1]], self.structure])
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
            plt.savefig('grid_searchResults/'+'SdA_structure_'+str(self.item)+'.png')

    
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
        plt.savefig('grid_searchResults/'+'softmax_train_'+str(self.item)+'.png')

     
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
        plt.savefig('grid_searchResults/'+'fine_tune_structure_'+str(self.item)+'.png')

     
    def predict(self, X):
        #print self self=sDA
        tmp = X
        for L in self.Layers:
            tmp = L.activate(tmp)
        return tmp
               
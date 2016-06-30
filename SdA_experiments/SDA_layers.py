import numpy  
import Noise
import Layers
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class Trainer():
    def train(self, layers, X, y, dA_avg_costs, dA_iter, epochs):
        rows = numpy.arange(X.shape[0])
        ## stochastic sampler in n_iterations
        for iters in range(epochs):
            numpy.random.shuffle(rows)
            for idx in rows:
                tmp = X[[idx]]
                for i in range(len(layers)):
                    tmp = layers[i].activate(tmp)
                grad, dA_avg_costs, dA_iter = layers[-1].backwardPass(y[[idx]], dA_avg_costs, dA_iter)
                for i in range(len(layers)-1, -1, -1):
                    grad = layers[i].update(grad)
        return layers, [dA_avg_costs, dA_iter]
        
        
class StackedDA():
    def __init__(self, structure, alpha=0.01):
        self.alpha = alpha
        self.structure = structure
        self.Layers = []
        print "Call \"pre_train(epochs)\""
        
    def __repr__(self):
        return "\nStacked Denoising Autoencoder Structure:\t%s"%(" -> ".join([str(x) for x in self.structure]))
        
    def pre_train(self, X, epochs=1, noise_rate=0.3):
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
            dA_avg_costs = []
            dA_iter = [] 
            layers, dA_avg_perm = trainer.train([s1, s2], self.X, self.X, dA_avg_costs, dA_iter, epochs)
            s1, s2 = layers
            self.X = s1.activate(self.X)
            self.Layers.append(s1)
            
            ##############
            # Format      
            #################           
            LLdata = [float(L) for L in dA_avg_perm[0]]
            LLiter = [float(it) for it in dA_avg_perm[1]]
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
            plt.savefig('grid_searchResults/'+'SdA_structure_'+str([self.structure])+'_noise_'+str(noise_rate)+'_pretrain_epochs_'+str(epochs)+'.png')

            
            
    
    def finalLayer(self, X, y, epochs=1, n_neurons=200):
        print "Final Layer" 
        V = self.predict(X)
        softmax = Layers.SoftmaxLayer(self.Layers[-1].W.shape[1], y.shape[1]) 
        
        #########################
        # Final layer of THE MODEL #
        #########################
        dA_avg_costs = []
        dA_iter = [] 
        softmax, dA_avg_perm = Trainer().train([softmax], V, y, dA_avg_costs, dA_iter, epochs)
        self.Layers.append(softmax[0])
        
        ##############
        # Format      
        #################           
        LLdata = [float(L) for L in dA_avg_perm[0]]
        LLiter = [float(it) for it in dA_avg_perm[1]]
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
        plt.title('softmax_n_neurons_'+str(n_neurons)+'_train_epochs_'+str(epochs), fontsize=9)
        plt.savefig('grid_searchResults/'+'softmax_n_neurons_'+str(n_neurons)+'_train_epochs_'+str(epochs)+'.png')

     
    def fine_tune(self, X, y, epochs=1):
        print "Fine Tunning" 
        #########################
        # Fine Tunning THE MODEL #
        #########################
        dA_avg_costs = []
        dA_iter = [] 
        self.Layers, dA_avg_perm = Trainer().train(self.Layers, X, y, dA_avg_costs, dA_iter, epochs)
        
        ##############
        # Format      
        #################           
        LLdata = [float(L) for L in dA_avg_perm[0]]
        LLiter = [float(it) for it in dA_avg_perm[1]]
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
               
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:03:49 2016

Helper functions for Deep learning

@author: Cristina Gallego

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
-  Based on code from: git clone https://github.com/lisa-lab/DeepLearningTutorials.git

"""

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import random

import numpy as np
import pandas as pd
import theano
import theano.tensor as T

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

from LogisticRegression import *
from MultilayerPerceptron import *
from dAutoencoder import *
from StackeddAutoencoder import *

from utils import tile_raster_images
try:
    import PIL.Image as Image
except ImportError:
    import Image


class DLFuncs_SdA(object):
    """DLFuncs_SdA Class 
    from DLFuncs_SdA import *
    funcs = DLFuncs_SdA()
    self=funcs
    """
    
    def __init__(self):
        self.ptcsize = None
        self.img_size = None
        

    def __call__(self):       
        """ Turn Class into a callable object """
        DeepLearnFuncs()
    
    def randomList(self, a): 
        b = [] 
        for i in range(len(a)): 
            element = random.choice(a) 
            a.remove(element) 
            b.append(element) 
        return b
    

    def shared_dataset(self, data_x, data_y, borrow=True):
        """ Function that loads the dataset into shared variables

        Allow Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        # or tuple data_x, data_y = data_xy        
        shared_x = theano.shared(np.asarray(data_x,dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
 
    
    def load_wUdata(self, traindata_path, labeldata_path, trainUdata_path)       :
        ''' Loads the datasets incuding Unlabeled data
    
        :type dataset: string
        :param data_path: the path to the dataset 
        '''
        data_dir, data_file = os.path.split(traindata_path)

        print('... loading data')

        # Load the dataset
        with gzip.open(traindata_path, 'rb') as f:
            try:
                datasets = pickle.load(f, encoding='latin1')
            except:
                datasets = pickle.load(f)
        
        # Load the Udataset
        with gzip.open(trainUdata_path, 'rb') as f:
            try:
                Udatasets = pickle.load(f, encoding='latin1')
            except:
                Udatasets = pickle.load(f)
    
        with gzip.open(labeldata_path, 'rb') as f:
            try:
                labels = pickle.load(f, encoding='latin1')
            except:
                labels = pickle.load(f)
        
        print(labels.describe()) 
        
        # Assign labels as 1 if Enhancement, 0 if not enhancement
        LandUlabels = np.concatenate( (np.ones(len(datasets)), np.zeros(len(Udatasets))) )
        Ulabels = np.zeros(len(Udatasets))
        LandUmergedata = []
        LandUmergedata = datasets + Udatasets

        #############
        ## RANDOMIZE data into 3 groups: Train, valid, train, by unique lesion_id  
        #############
        uids = np.unique(labels['lesion_id'])
        randuids = self.randomList(list(uids))
        
        lentrain = int(round(len(randuids)*0.6))
        lenvalid = int(round(len(randuids)*0.2))
        lentest = int(round(len(randuids)*0.2))
        
        # split ids based on random choice in 60% train, 20% valid and 20% test
        idstrain = [randuids[i] for i in range(lentrain)]
        idsvalid = [randuids[i] for i in range(lentrain,(lentrain+lenvalid))]
        idstest = [randuids[i] for i in range(lentrain+lenvalid,len(randuids))]
        
        
        randuidsU = self.randomList(list(range(len(Udatasets))))
        lentrainU = int(round(len(randuidsU)*0.6))
        lenvalidU = int(round(len(randuidsU)*0.2))
        lentestU = int(round(len(randuidsU)*0.2))
        
        # split ids based on random choice in 60% train, 20% valid and 20% test
        inxstrain = randuidsU[0:lentrainU]
        inxvalid = randuidsU[lentrainU:lentrainU+lenvalidU]
        inxtest = randuidsU[lentrainU+lenvalidU::] 

        # make sure data_x elements are all 30*30*4 = 3600 if not remove patch
        ptsize = LandUmergedata[0].shape[0]
        # find index element different from ptsize
        sizes=[(k,elem.shape[0]) for elem,k in zip(LandUmergedata,range(len(LandUmergedata))) if elem.shape[0] != ptsize]
        # mark those indexes to exclude        
        remindx = [int(pindx[0]) for pindx in sizes]        
        print("Removing incomplete patches...")
        print(remindx)
        
        # assign each id to bucket
        traindata = []; validdata = []; testdata = [];
        trainlabel = []; validlabel = []; testlabel = []; 
        
        # process Labeled data
        for i in range(len(labels)):
            if i not in remindx:
                if labels.iloc[i]['lesion_id'] in idstrain:
                    traindata.append(datasets[i])
                    trainlabel.append(LandUlabels[i])
                
                if labels.iloc[i]['lesion_id'] in idsvalid:
                    validdata.append(datasets[i])
                    validlabel.append(LandUlabels[i])
                    
                if labels.iloc[i]['lesion_id'] in idstest:
                    testdata.append(datasets[i])
                    testlabel.append(LandUlabels[i])
                    
        # process Un-Labeled data
        for i in range(len(Udatasets)):
            if i in inxstrain:
                traindata.append(Udatasets[i])
                trainlabel.append(Ulabels[i])
            
            if i in inxvalid:
                validdata.append(Udatasets[i])
                validlabel.append(Ulabels[i])
                
            if i in inxtest:
                testdata.append(Udatasets[i])
                testlabel.append(Ulabels[i])

        #############
        # CONVERT to theano shared vars
        #############
        train_set_x, train_set_y = self.shared_dataset(traindata, trainlabel)
        valid_set_x, valid_set_y = self.shared_dataset(validdata, validlabel)
        test_set_x, test_set_y = self.shared_dataset(testdata, testlabel)
    
        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
            
        return rval
        

        
    def load_data(self, traindata_path, labeldata_path):
        ''' Loads the dataset
    
        :type dataset: string
        :param data_path: the path to the dataset 
        '''
        #############
        # LOAD DATA 
        # eg. traindata_path='Z://Cristina//Section3//DL_space//allLpatches.pklz'
        # eg. labeldata_path='Z://Cristina//Section3//DL_space//allLabels.pklz'
        #############
        data_dir, data_file = os.path.split(traindata_path)

        print('... loading data')

        # Load the dataset
        with gzip.open(traindata_path, 'rb') as f:
            try:
                datasets = pickle.load(f, encoding='latin1')
            except:
                datasets = pickle.load(f)
    
        with gzip.open(labeldata_path, 'rb') as f:
            try:
                labels = pickle.load(f, encoding='latin1')
            except:
                labels = pickle.load(f)
        
        print(labels.describe())  
        
        # train_set, valid_set, test_set format: tuple(input, target)
        # input is a numpy.ndarray of 2 dimensions (a matrix)
        # where each row corresponds to an example. target is a
        # numpy.ndarray of 1 dimension (vector) that has the same length as
        # the number of rows in the input. It should give the target
        # to the example with the same index in the input.
                
        #############
        # FORMAT labels into integers
        #############
        # process label for nmenh distributions
        # e.g [(u'Focal', 0), (u'Linear', 1), (u'N/A', 2), (u'Regional', 3), (u'Segmental', 4)]
        # recode each case in distLabel with integers
        codesL = zip(np.unique(labels['nmenh_dist']), range(len(np.unique(labels['nmenh_dist']))))
        print(codesL)
        
        distCodes = [distN[1] for distN in codesL]
        distLabel = []
        for k in range(len(labels)):
            label = labels.iloc[k]['nmenh_dist'] 
            sellabel = [distN[0]==label for distN in codesL]
            code = [i for (i, v) in zip(distCodes, sellabel) if v][0]
            distLabel.append(code)
            
        #############
        ## RANDOMIZE data into 3 groups: Train, valid, train, by unique lesion_id  
        #############
        uids = np.unique(labels['lesion_id'])
        randuids = self.randomList(list(uids))
        
        lentrain = int(round(len(randuids)*0.6))
        lenvalid = int(round(len(randuids)*0.2))
        lentest = int(round(len(randuids)*0.2))
        
        # split ids based on random choice in 60% train, 20% valid and 20% test
        idstrain = [randuids[i] for i in range(lentrain)]
        idsvalid = [randuids[i] for i in range(lentrain,(lentrain+lenvalid))]
        idstest = [randuids[i] for i in range(lentrain+lenvalid,len(randuids))]
            
        # make sure data_x elements are all 30*30*4 = 3600 if not remove patch
        ptsize = datasets[0].shape[0]
        # find index element different from ptsize
        sizes=[(k,elem.shape[0]) for elem,k in zip(datasets,range(len(datasets))) if elem.shape[0] != ptsize]
        # mark those indexes to exclude        
        remindx = [int(pindx[0]) for pindx in sizes]        
        print("Removing incomplete patches...")
        print(remindx)
        
        # assign each id to bucket
        traindata = []; validdata = []; testdata = [];
        trainlabel = []; validlabel = []; testlabel = []; 
        for i in range(len(labels)):
            if i not in remindx:
                if labels.iloc[i]['lesion_id'] in idstrain:
                    traindata.append(datasets[i])
                    trainlabel.append(distLabel[i])
                
                if labels.iloc[i]['lesion_id'] in idsvalid:
                    validdata.append(datasets[i])
                    validlabel.append(distLabel[i])
                    
                if labels.iloc[i]['lesion_id'] in idstest:
                    testdata.append(datasets[i])
                    testlabel.append(distLabel[i])

        ##############
        ## FOR 2D space only paches, reshape and extract only a time point 
        ## e.g 1st post contrast
        ##############
        traindata1st = [ datum.reshape(4,900)[0,:] for datum in traindata  if datum.shape[0]  == ptsize]
        validdata1st = [ datum.reshape(4,900)[0,:] for datum in validdata  if datum.shape[0]  == ptsize]
        testdata1st = [ datum.reshape(4,900)[0,:] for datum in testdata  if datum.shape[0]  == ptsize]

        # test example patches
        fig, axes = plt.subplots(ncols=3, nrows=1)
        axes[0].imshow(traindata1st[0].reshape(30,30), cmap="Greys_r")
        axes[1].imshow(validdata1st[0].reshape(30,30), cmap="Greys_r")
        axes[2].imshow(testdata1st[0].reshape(30,30), cmap="Greys_r")
        plt.show()   
        
        print 'Number of patches: %i train, %i valid, %i test' % (len(traindata1st),len(validdata1st),len(testdata1st))
        # plot train
        stringlabels = ["Focal" if l == 0 else "Linear" if l == 1 else "Regional" if l == 3 else "Segmental" if l == 4 else "N/A" if l == 2 else l for l in trainlabel]
        dftrainlabel = pd.DataFrame( trainlabel )
        dftrainlabel.columns = ['EnhCode']
        dftrainlabel['EnhType'] = stringlabels
        # plot train
        fig, axes = plt.subplots(ncols=1, nrows=3)
        sns.countplot(y="EnhType", hue="EnhType", data=dftrainlabel, ax=axes[0])
        
        # plot valid
        stringlabels = ["Focal" if l == 0 else "Linear" if l == 1 else "Regional" if l == 3 else "Segmental" if l == 4 else "N/A" if l == 2 else l for l in validlabel]
        dfvalidlabel = pd.DataFrame( validlabel )
        dfvalidlabel.columns = ['EnhCode']
        dfvalidlabel['EnhType'] = stringlabels
        sns.countplot(y="EnhType", hue="EnhType", data=dfvalidlabel, ax=axes[1])

        # plot test
        stringlabels = ["Focal" if l == 0 else "Linear" if l == 1 else "Regional" if l == 3 else "Segmental" if l == 4 else "N/A" if l == 2 else l for l in testlabel]
        dftestlabel = pd.DataFrame( testlabel )
        dftestlabel.columns = ['EnhCode']
        dftestlabel['EnhType'] = stringlabels
        sns.countplot(y="EnhType", hue="EnhType", data=dftestlabel, ax=axes[2])

        # save the best model
        with open('test_labels.pkl', 'wb') as f:
            pickle.dump(testlabel, f)

        #############
        # CONVERT to theano shared vars
        #############
        train_set_x, train_set_y = self.shared_dataset(traindata1st, trainlabel)
        valid_set_x, valid_set_y = self.shared_dataset(validdata1st, validlabel)
        test_set_x, test_set_y = self.shared_dataset(testdata1st, testlabel)
    
        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
            
        return rval
        

    def test_SdA_timep(self, pretraining_epochs, pretrain_lr, batch_size,
                        training_epochs, finetune_lr,  
                        corruption_levels, 
                        hidden_layers_sizes, hidden_layers_sidelen, output_folder):
        """
        Demonstrates how to train and test a stochastic denoising autoencoder.
        
        :type learning_rate: float
        :param learning_rate: learning rate used in the finetune stage
        (factor for the stochastic gradient)
    
        :type pretraining_epochs: int
        :param pretraining_epochs: number of epoch to do pretraining
    
        :type pretrain_lr: float
        :param pretrain_lr: learning rate to be used during pre-training
    
        :type n_iter: int
        :param n_iter: maximal number of iterations ot run the optimizer
    
        :type dataset: string
        :param dataset: path the the pickled dataset
    
        """
    
        traindata_path='Z://Cristina//Section3//SdA_experiments//allLpatches.pklz'
        trainUdata_path='Z://Cristina//Section3//SdA_experiments//allUpatches.pklz'
        labeldata_path='Z://Cristina//Section3//SdA_experiments//allLabels.pklz'
        
        #############
        ## LOAD datasets
        #############
        datasets = self.load_wUdata(traindata_path, labeldata_path, trainUdata_path)
        
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
    
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches //= batch_size
    
        # numpy random generator
        numpy_rng = np.random.RandomState(89677)
        
        print('... building the Stacked Autoenconder model')
        
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        os.chdir(output_folder)
        
        # construct the stacked denoising autoencoder class
        sda = SdA(
            numpy_rng=numpy_rng,
            n_ins = 30*30,
            hidden_layers_sizes=hidden_layers_sizes,
            corruption_levels=corruption_levels,
            n_outs=2
        )

        #########################
        # PRETRAINING THE MODEL #
        #########################
        dA_avg_costs = []
        dA_iter = []
        layer_i = []
        
        print('... getting the pretraining functions')
        pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                    batch_size=batch_size)
    
        print('... pre-training the model')
        start_time = timeit.default_timer()
        
        ## Pre-train layer-wise
        for i in range(sda.n_layers):
            # go through pretraining epochs
            for epoch in range(pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in range(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index,
                             corruption=corruption_levels[i],
                             lr=pretrain_lr))
            
                # append      
                dA_avg_costs.append(  np.mean(c) )
                dA_iter.append(epoch)
                layer_i.append(i)
                print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, np.mean(c)))
    
            #####################################
            # Plot images in 2D
            #####################################   
            Xtmp = sda.dA_layers[i].W.get_value(borrow=True).T
            imgX = Xtmp.reshape( Xtmp.shape[0], hidden_layers_sidelen[i], hidden_layers_sidelen[i])
            image = Image.fromarray(
                tile_raster_images(X=imgX , img_shape=(hidden_layers_sidelen[i], hidden_layers_sidelen[i]), 
                                   tile_shape=(10, 10),
                                   tile_spacing=(1, 1)))

            #show and save                     
            image.save('filters_corruption_layer_'+str(i)+'_'+str(float(corruption_levels[i]))+'.png')
            # prepare display    
            fig, ax = plt.subplots()  
            ax.imshow(image,  cmap="Greys_r")
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            
        os.chdir('../../')
        
        end_time = timeit.default_timer()
        print(('The pretraining code for file ' +
            os.path.split(__file__)[1] +
                ' ran for %.2fm' % ((end_time - start_time) / 60.)))

        
        ###############
        ## Visualize second layer filter by Lee et al. method
        ###############
        
        
        

        ##############
        # Format      
        #################           
        LLdata = [float(L) for L in dA_avg_costs]
        LLiter = [float(it) for it in dA_iter]
        LLilayer = [ilayer for ilayer in layer_i]
        dfpredata = pd.DataFrame( LLdata )
        dfpredata.columns = ['LL_iter']
        dfpredata['iter'] = LLiter
        dfpredata['layer'] = LLilayer
        
        
        ########################
        # FINETUNING THE MODEL #
        ########################
        # get the training, validation and testing function for the model
        print('... getting the finetuning functions')
        train_fn, validate_model, test_model = sda.build_finetune_functions(
            datasets=datasets,
            batch_size=batch_size,
            learning_rate=finetune_lr
        )
    
        print('... finetunning the Stacked model')
        ############
        ### for plotting likelihood or cost, accumulate returns of train_model
        ############
        minibatch_avg_costs = []
        minibatch_iter = []
        minibatch_loss = []
        
        # early-stopping parameters
        patience = 100 * n_train_batches  # look as this many examples regardless
        patience_increase = 2.  # wait this much longer when a new best is
                                # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience // 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch
    
        best_validation_loss = numpy.inf
        test_score = 0.
        start_time = timeit.default_timer()
    
        done_looping = False
        epoch = 0
    
        while (epoch < training_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):
                
                minibatch_avg_cost = train_fn(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index
    
                if (iter + 1) % validation_frequency == 0:
                    validation_losses = validate_model()
                    this_validation_loss = np.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, 
                           n_train_batches,
                           this_validation_loss * 100.))
                           
                    ##############
                    # append      
                    #################
                    minibatch_avg_costs.append(minibatch_avg_cost)
                    minibatch_iter.append(iter)
                    minibatch_loss.append(this_validation_loss*100)
    
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
    
                        #improve patience if loss improvement is good enough
                        if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)
    
                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter
    
                        # test it on the test set
                        test_losses = test_model()
                        test_score = np.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))
    
                if patience <= iter:
                    done_looping = True
                    break
    
        end_time = timeit.default_timer()
        print 'Optimization complete with best validation score of %f, on iteration %i, with test performance %f' % ((best_validation_loss * 100.), (best_iter + 1), (test_score * 100.))
    
        print(('The training code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)))
               
        ##############
        # Format      
        #################           
        LLdata = [float(L) for L in minibatch_avg_costs]
        LLiter = [float(it) for it in minibatch_iter]
        LLoss = [float(l) for l in minibatch_loss]        
        dfinedata = pd.DataFrame( LLdata )
        dfinedata.columns = ['LL_iter']
        dfinedata['iter'] = LLiter
        dfinedata['loss'] = LLoss
    
        return [dfpredata, dfinedata]
    
    
 
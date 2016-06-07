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
from LeNetConvPoolLayer import *
from dAutoencoder import *
from StackeddAutoencoder import *
from RBM import *
from DBN import *

from utils import tile_raster_images
try:
    import PIL.Image as Image
except ImportError:
    import Image


class DeepLearnFuncs(object):
    """DeepLearnFuncs Class 
    from DeepLearnFuncs import *
    funcs = DeepLearnFuncs()
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
        
        
    def sgd_optimization(self, learning_rate, n_epochs, batch_size):
        """
        Stochastic gradient descent optimization of a log-linear model
    
        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)
    
        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer
    
        :type dataset: string
        
        """
        traindata_path='Z://Cristina//Section3//DL_space//allLpatches.pklz'
        labeldata_path='Z://Cristina//Section3//DL_space//allLabels.pklz'
        
        #############
        ## LOAD datasets
        #############
        datasets = self.load_data(traindata_path, labeldata_path)
    
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
        
        # save the best model
        with open('test_set.pkl', 'wb') as f:
            pickle.dump(datasets[2], f)
            
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
        
        
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('... building a LogReg model')
        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
    
        # generate symbolic variables for input (x and y represent a
        # minibatch)
        x = T.matrix('x')  # data, presented as rasterized images
        y = T.ivector('y')  # labels, presented as 1D vector of [int] labels
    
        # construct the logistic regression class
        # Each image has size 30*30*4 = 3600 and 6 classes
        # Classes: [(u'Ductal', 0), (u'Focal', 1), (u'Linear', 2), (u'N/A', 3), (u'Regional', 4), (u'Segmental', 5)]
        classifier = LogisticRegression(input=x, n_in=900, n_out=6)
    
        # the cost we minimize during training is the negative log likelihood of
        # the model in symbolic format
        cost = classifier.negative_log_likelihood(y)
    
        # compiling a Theano function that computes the mistakes that are made by
        # the model on a minibatch
        test_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
    
        validate_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
    
        # compute the gradient of cost with respect to theta = (W,b)
        g_W = T.grad(cost=cost, wrt=classifier.W)
        g_b = T.grad(cost=cost, wrt=classifier.b)
    
        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs.
        updates = [(classifier.W, classifier.W - learning_rate * g_W),
                   (classifier.b, classifier.b - learning_rate * g_b)]
    
        # compiling a Theano function `train_model` that returns the cost, but in
        # the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
        
        ###############
        # TRAIN MODEL #
        ###############
        print('... training the model n_train_batches = %d' % n_train_batches)
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                                      # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                      # considered significant
        validation_frequency = min(n_train_batches, patience // 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch
    
        best_validation_loss = np.inf
        test_score = 0.
        start_time = timeit.default_timer()
    
        done_looping = False
        epoch = 0

        ############
        ### for plotting likelihood or cost, accumulate returns of train_model
        ############
        minibatch_avg_costs = []
        minibatch_iter = []
        minibatch_loss = []
        
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):
    
                minibatch_avg_cost = train_model(minibatch_index)
                
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index
    
                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i)
                                         for i in range(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)
    
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (   epoch,
                            (minibatch_index + 1),
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )

                    ##############
                    # append      
                    #################
                    minibatch_avg_costs.append(minibatch_avg_cost)
                    minibatch_iter.append(iter)
                    minibatch_loss.append(this_validation_loss*100)
    
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)
    
                        best_validation_loss = this_validation_loss
                        # test it on the test set
    
                        test_losses = [test_model(i)
                                       for i in range(n_test_batches)]
                        test_score = np.mean(test_losses)
    
                        print(('epoch %i, minibatch %i/%i, test error of'
                                ' best model %f %%') %
                            (
                                epoch,
                                (minibatch_index + 1),
                                n_train_batches,
                                test_score * 100.
                            ))
    
                        # save the best model
                        with open('best_model.pkl', 'wb') as f:
                            pickle.dump(classifier, f)
                            
    
                if patience <= iter:
                    done_looping = True
                    break
    
        end_time = timeit.default_timer()
        
        
        print('Optimization complete with best validation score of %f %%,'
                'with test performance %f %%'
            % (best_validation_loss * 100., test_score * 100.) )
        print('The code run for %d epochs, with %f epochs/sec' 
            % (epoch, 1. * epoch / (end_time - start_time)))
        print('The code for file ' + os.path.split(__file__)[1] +
               ' ran for %.1fs' % (end_time - start_time))
               
        ##############
        # Format      
        #################           
        LLdata = [float(L) for L in minibatch_avg_costs]
        LLiter = [float(i) for i in minibatch_iter]
        LLoss = [float(l) for l in minibatch_loss]
        dfLLdata = pd.DataFrame( LLdata )
        dfLLdata.columns = ['LL_iter']
        dfLLdata['iter'] = LLiter
        dfLLdata['0-1-loss'] = LLoss
        
        return dfLLdata
        
        
               
    def test_mlp(self, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
                 batch_size=20, n_hidden=500):
        """
        stochastic gradient descent optimization for a multilayer perceptron
    
        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
        gradient
    
        :type L1_reg: float
        :param L1_reg: L1-norm's weight when added to the cost (see
        regularization)
    
        :type L2_reg: float
        :param L2_reg: L2-norm's weight when added to the cost (see
        regularization)
    
        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer
    

        """
        traindata_path='Z://Cristina//Section3//DL_space//allLpatches.pklz'
        labeldata_path='Z://Cristina//Section3//DL_space//allLabels.pklz'
        
        #############
        ## LOAD datasets
        #############
        datasets = self.load_data(traindata_path, labeldata_path)
    
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
        
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('... building the MLP model, learning rate %f' % learning_rate)
    
        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        x = T.matrix('x')  # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels
    
        rng = np.random.RandomState(1234)
    
        # construct the MLP class
        classifier = MLP(
            rng=rng,
            input=x,
            n_in=30*30,
            n_hidden=n_hidden,
            n_out=6
        )
    
        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically
        cost = (
            classifier.negative_log_likelihood(y)
            + L1_reg * classifier.L1
            + L2_reg * classifier.L2_sqr
        )
    
        # compiling a Theano function that computes the mistakes that are made
        # by the model on a minibatch
        test_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]
            }
        )
    
        validate_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]
            }
        )
    
        # compute the gradient of cost with respect to theta (sotred in params)
        # the resulting gradients will be stored in a list gparams
        gparams = [T.grad(cost, param) for param in classifier.params]
    
        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs
    
        # given two lists of the same length, A = [a1, a2, a3, a4] and
        # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
        # element is a pair formed from the two lists :
        #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(classifier.params, gparams)
        ]
    
        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
    
        ###############
        # TRAIN MODEL #
        ###############
        print('... training')
    
        # early-stopping parameters
        patience = 100000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience // 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch
    
        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()
    
        epoch = 0
        done_looping = False
        
        ############
        ### for plotting likelihood or cost, accumulate returns of train_model
        ############
        minibatch_avg_costs = []
        minibatch_iter = []
        minibatch_loss = []
    
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):
    
                minibatch_avg_cost = train_model(minibatch_index)
                
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index
    
                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in range(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)
    
                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )
                    
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
    
                        best_validation_loss = this_validation_loss
                        best_iter = iter
    
                        # test it on the test set
                        test_losses = [test_model(i) for i
                                       in range(n_test_batches)]
                        test_score = np.mean(test_losses)
    
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))
                               
                        # save the best model
                        with open('best_modelMLP.pkl', 'wb') as f:
                            pickle.dump(classifier.logRegressionLayer, f)
                               
    
                if patience <= iter:
                    done_looping = True
                    break
    
        end_time = timeit.default_timer()
        
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
              
        print('The code run for %d epochs, with %f epochs/sec' 
            % (epoch, 1. * epoch / (end_time - start_time)))
        print('The code for file ' + os.path.split(__file__)[1] +
               ' ran for %.1fs' % (end_time - start_time))
               
        ##############
        # Format      
        #################           
        LLdata = [float(L) for L in minibatch_avg_costs]
        LLiter = [float(i) for i in minibatch_iter]
        LLoss = [float(l) for l in minibatch_loss]
        dfLLdata = pd.DataFrame( LLdata )
        dfLLdata.columns = ['LL_iter']
        dfLLdata['iter'] = LLiter
        dfLLdata['0-1-loss'] = LLoss
        
        return dfLLdata   
        


    def evaluate_lenet5(self, learning_rate=0.1, n_epochs=200,
                        nkerns=[20, 50], batch_size=500):
                            
        """ Demonstrates lenet
    
        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)
    
        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer
    
        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer
        """
    
        rng = np.random.RandomState(23455)
    
        traindata_path='Z://Cristina//Section3//DL_space//allLpatches.pklz'
        labeldata_path='Z://Cristina//Section3//DL_space//allLabels.pklz'
        
        #############
        ## LOAD datasets
        #############
        datasets = self.load_data(traindata_path, labeldata_path)
    
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
        
        tx = train_set_x.get_value(borrow=True)
    
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_train_batches //= batch_size
        n_valid_batches //= batch_size
        n_test_batches //= batch_size
    
        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
    
        # start-snippet-1
        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels
    
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('... building the LeNet5 model ')
    
        # Reshape matrix of rasterized images of shape (batch_size, 4*30*30)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        layer0_input = x.reshape((batch_size, 4, 30, 30))
    
        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (30-7+1 , 30-7+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
        layer0 = LeNetConvPoolLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 4, 30, 30),
            filter_shape=(nkerns[0], 4,5, 5),
            poolsize=(2, 2)
        )
    
        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2)
        )
    
        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        layer2_input = layer1.output.flatten(2)
    
        # construct a fully-connected sigmoidal layer
        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in=nkerns[1] * 4 * 4,
            n_out=1000,
            activation=T.tanh
        )
    
        # classify the values of the fully-connected sigmoidal layer
        layer3 = LogisticRegression(input=layer2.output, n_in=1000, n_out=6)
    
        # the cost we minimize during training is the NLL of the model
        cost = layer3.negative_log_likelihood(y)
    
        # create a function to compute the mistakes that are made by the model
        test_model = theano.function(
            [index],
            layer3.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
    
        validate_model = theano.function(
            [index],
            layer3.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
    
        # create a list of all model parameters to be fit by gradient descent
        params = layer3.params + layer2.params + layer1.params + layer0.params
    
        # create a list of gradients for all model parameters
        grads = T.grad(cost, params)
    
        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
        ]
    
        train_model = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
    
        ###############
        # TRAIN MODEL #
        ###############
        print('... training')
        # early-stopping parameters
        patience = 100000 # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience // 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch
    
        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()
    
        epoch = 0
        done_looping = False

        ############
        ### for plotting likelihood or cost, accumulate returns of train_model
        ############
        minibatch_avg_costs = []
        minibatch_iter = []
        minibatch_loss = []
        
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):
                
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index
    
                # every 100 iterations train the model
                if iter % 100 == 0:
                    print('training @ iter = ', iter)
                    cost_ij = train_model(minibatch_index)
    
                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in range(n_valid_batches)]
                                         
                    this_validation_loss = np.mean(validation_losses)
                    
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))

                    ##############
                    # append      
                    #################
                    minibatch_avg_costs.append(cost_ij)
                    minibatch_iter.append(iter)
                    minibatch_loss.append(this_validation_loss*100)   
                    
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
    
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)
    
                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter
    
                        # test it on the test set
                        test_losses = [test_model(i) for i in range(n_test_batches)]
                        test_score = np.mean(test_losses)
                        
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))
    
                if patience <= iter:
                    done_looping = True
                    break
    
        end_time = timeit.default_timer()
        
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print('The code for file ' + os.path.split(__file__)[1] +
               ' ran for %.1fs' % (end_time - start_time))
               
        ##############
        # Format      
        #################           
        LLdata = [float(L) for L in minibatch_avg_costs]
        LLiter = [float(i) for i in minibatch_iter]
        LLoss = [float(l) for l in minibatch_loss]
        dfLLdata = pd.DataFrame( LLdata )
        dfLLdata.columns = ['LL_iter']
        dfLLdata['iter'] = LLiter
        dfLLdata['0-1-loss'] = LLoss
        
        return dfLLdata  
        
        
    def test_dA(self, learning_rate=0.1, training_epochs=15, corruptionL=0.0,
                batch_size=20, output_folder='dA_plots'):
    
        """
        :type learning_rate: float
        :param learning_rate: learning rate used for training the DeNosing
                              AutoEncoder
    
        :type training_epochs: int
        :param training_epochs: number of epochs used for training
    
        """
        traindata_path='Z://Cristina//Section3//DL_space//allLpatches.pklz'
        labeldata_path='Z://Cristina//Section3//DL_space//allLabels.pklz'
        
        #############
        ## LOAD datasets
        #############
        datasets = self.load_data(traindata_path, labeldata_path)
        train_set_x, train_set_y = datasets[0]
    
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        os.chdir(output_folder)
 
        ####################################
        # BUILDING THE MODEL NO CORRUPTION #
        ####################################
        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch
        x = T.matrix('x')  # the data is presented as rasterized images
    
        rng = np.random.RandomState(123)
        theano_rng = RandomStreams(rng.randint(2 ** 30))
    
        da = dA(
            numpy_rng=rng,
            theano_rng=theano_rng,
            input=x,
            n_visible=30*30,
            n_hidden=100
        )
    
        cost, updates = da.get_cost_updates(
            corruption_level=float(corruptionL),
            learning_rate=learning_rate
        )
    
        train_da = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size]
            }
        )
    
        start_time = timeit.default_timer()
                    
        ############
        # TRAINING #
        ############
        minibatch_avg_costs = []
        minibatch_iter = []
        
        # go through training epochs
        for epoch in range(training_epochs):
            # go through trainng set
            c = []
            for batch_index in range(n_train_batches):
                c.append(train_da(batch_index))
            
            # append      
            minibatch_avg_costs.append(  np.mean(c) )
            minibatch_iter.append(epoch)
            print('Training epoch %d, cost ' % epoch, np.mean(c))
    
        print('The %f corruption dAutoenconder finished ' % float(corruptionL))
                              
        #####################################
        # Plot images in 2D
        #####################################   
        Xtmp = da.W.get_value(borrow=True).T
        imgX = Xtmp.reshape( Xtmp.shape[0], 30, 30)
        image = Image.fromarray(
            tile_raster_images(X=imgX , img_shape=(30, 30), tile_shape=(10, 10),
                               tile_spacing=(1, 1)))
                               
        image.save('filters_corruption_'+str(float(corruptionL))+'.png')
        
        # prepare display    
        fig, ax = plt.subplots()  
        ax.imshow(image,  cmap="Greys_r")
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        ##############
        # Format      
        #################           
        LLdata = [float(L) for L in minibatch_avg_costs]
        LLiter = [float(i) for i in minibatch_iter]
        dfLLdata = pd.DataFrame( LLdata )
        dfLLdata.columns = ['LL_iter']
        dfLLdata['iter'] = LLiter
        
        os.chdir('../../')
        
        return dfLLdata  
        

    def test_SdA(self, finetune_lr=0.1, pretraining_epochs=15,
             pretrain_lr=0.001, training_epochs=1000,
             batch_size=1, corruption_levels=[0.35, 0.35, 0.1], output_folder= 'plots/SdA_plots'):
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
    
        traindata_path='Z://Cristina//Section3//DL_space//allLpatches.pklz'
        labeldata_path='Z://Cristina//Section3//DL_space//allLabels.pklz'
        
        #############
        ## LOAD datasets
        #############
        datasets = self.load_data(traindata_path, labeldata_path)
        
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
            hidden_layers_sizes=[625, 625, 625],
            corruption_levels=[0.35, 0.15, 0.1],
            n_outs=6
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
            if i == 0:
                imgX = Xtmp.reshape( Xtmp.shape[0], 30, 30)
                image = Image.fromarray(
                    tile_raster_images(X=imgX , img_shape=(30, 30), tile_shape=(10, 10),
                                       tile_spacing=(1, 1)))
            else:
                imgX = Xtmp.reshape( Xtmp.shape[0], 25, 25)
                image = Image.fromarray(
                    tile_raster_images(X=imgX , img_shape=(25, 25), tile_shape=(10, 10),
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
        W1=sda.dA_layers[0].W.get_value(borrow=True).T
        W2=sda.dA_layers[1].W.get_value(borrow=True).T
        plt.imshow(W2, cmap="Greys_r")
        #W1=W1.reshape(100,30,30)
        
        Xtests_batches = test_set_x.get_value(borrow=True)
        aXtest = Xtests_batches[0,:]
        plt.imshow(aXtest.reshape(30,30), cmap="Greys_r")
        
        argSig = np.dot(W1, aXtest)
        plt.imshow(argSig.reshape(25,25), cmap="Greys_r")

        def sigmoid(x):
          return 1 / (1 + np.exp(-x))
          
        his = []
        hiSig = sigmoid(argSig)
        plt.imshow(hiSig.reshape(25,25), cmap="Greys_r")

        fig, ax = plt.subplots(ncols=25,nrows=25)
        k=0
        for i in range(25):
            for j in range(25):
                his.append(np.dot(np.transpose(W2[k,:]),hiSig))
                # weight the correspoding layer 2 filter
                imgW2 = np.dot(np.transpose(W2[k,:]),hiSig)*W2[k,:]
                ax[i,j].imshow(imgW2.reshape(25,25), cmap="Greys_r")
                ax[i,j].axes.get_xaxis().set_visible(False)
                ax[i,j].axes.get_yaxis().set_visible(False)
                k+=1
        
        dfhs = pd.DataFrame( his )
        dfhs.columns = ['his']
        dfhs.describe()

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
    
    
    
    def test_rbm(self, learning_rate=0.1, training_epochs=15,
             batch_size=20,
             n_chains=20, n_samples=10, output_folder='rbm_plots',
             n_hidden=500):
        """
        Demonstrate how to train and afterwards sample from it using Theano.
        
        :param learning_rate: learning rate used for training the RBM
    
        :param training_epochs: number of epochs used for training
        
        :param batch_size: size of a batch used to train the RBM
    
        :param n_chains: number of parallel Gibbs chains to be used for sampling
    
        :param n_samples: number of samples to plot for each chain
    
        """
        traindata_path='Z://Cristina//Section3//DeepLearning//allLpatches.pklz'
        labeldata_path='Z://Cristina//Section3//DeepLearning//allLabels.pklz'
        
        #############
        ## LOAD datasets
        #############
        datasets = self.load_data(traindata_path, labeldata_path)
    
        train_set_x, train_set_y = datasets[0]
        test_set_x, test_set_y = datasets[2]
    
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    
        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch
        x = T.matrix('x')  # the data is presented as rasterized images
    
        rng = np.random.RandomState(123)
        theano_rng = RandomStreams(rng.randint(2 ** 30))
    
        # initialize storage for the persistent chain (state = hidden
        # layer of chain)
        persistent_chain = theano.shared(np.zeros((batch_size, n_hidden),
                                                     dtype=theano.config.floatX),
                                         borrow=True)
    
        # construct the RBM class
        rbm = RBM(input=x, n_visible=4*30*30,
                  n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)
    
        # get the cost and the gradient corresponding to one step of CD-15
        cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                             persistent=persistent_chain, k=15)
    
        #################################
        #     Training the RBM          #
        #################################
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        os.chdir(output_folder)
    
        # it is ok for a theano function to have no output
        # the purpose of train_rbm is solely to update the RBM parameters
        train_rbm = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size]
            },
            name='train_rbm'
        )
    
        plotting_time = 0.
        start_time = timeit.default_timer()
    
        # go through training epochs
        for epoch in range(training_epochs):
            # go through the training set
            mean_cost = []
            for batch_index in range(n_train_batches):
                mean_cost += [train_rbm(batch_index)]
    
            print('Training epoch %d, cost is ' % epoch, np.mean(mean_cost))
    
            # Plot filters after each training epoch
            plotting_start = timeit.default_timer()
            
            # Construct image from the weight matrix
            Xtmp = rbm.W.get_value(borrow=True).T
            imgX = Xtmp.reshape( Xtmp.shape[0], 4, 30, 30)
            imgX0 = imgX[:,0,:,:]
            image = Image.fromarray(
                tile_raster_images(X=imgX0 , img_shape=(30, 30), tile_shape=(10, 10),
                                   tile_spacing=(1, 1)))
                                   
           
            image.save('filters_at_epoch_%i.png' % epoch)
            # prepare display    
            fig, ax = plt.subplots()  
            ax.imshow(image,  cmap="Greys_r")
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            
            plotting_stop = timeit.default_timer()
            plotting_time += (plotting_stop - plotting_start)
    
        end_time = timeit.default_timer()
    
        pretraining_time = (end_time - start_time) - plotting_time
    
        print ('Training took %f minutes' % (pretraining_time / 60.))


        #################################
        #     Sampling from the RBM     #
        #################################
        # find out the number of test samples
        number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]
    
        # pick random test examples, with which to initialize the persistent chain
        test_idx = rng.randint(number_of_test_samples - n_chains)
        persistent_vis_chain = theano.shared(
            np.asarray(
                test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
                dtype=theano.config.floatX
            )
        )
        
        plot_every = 1000
        # define one step of Gibbs sampling (mf = mean-field) define a
        # function that does `plot_every` steps before returning the
        # sample for plotting
        (
            [   presig_hids,
                hid_mfs,
                hid_samples,
                presig_vis,
                vis_mfs,
                vis_samples
            ],
            updates
        ) = theano.scan(
            rbm.gibbs_vhv,
            outputs_info=[None, None, None, None, None, persistent_vis_chain],
            n_steps=plot_every
        )
    
        # add to updates the shared variable that takes care of our persistent
        # chain :.
        updates.update({persistent_vis_chain: vis_samples[-1]})
        # construct the function that implements our persistent chain.
        # we generate the "mean field" activations for plotting and the actual
        # samples for reinitializing the state of our persistent chain
        sample_fn = theano.function(
            [],
            [
                vis_mfs[-1],
                vis_samples[-1]
            ],
            updates=updates,
            name='sample_fn'
        )
    
        # create a space to store the image for plotting ( we need to leave
        # room for the tile_spacing as well)
        image_data = np.zeros(
            (31 * n_samples + 1, 31 * n_chains - 1),
            dtype='uint8'
        )
        for idx in range(n_samples):
            # generate `plot_every` intermediate samples that we discard,
            # because successive samples in the chain are too correlated
            vis_mf, vis_sample = sample_fn()
            
            # format
            avis_mf = vis_mf.reshape(n_chains, 4, 900)
            aimg = avis_mf[:,0,:]
            
            print(' ... plotting sample %d' % idx)
            image_data[30 * idx:30 * idx + 30, :] = tile_raster_images(
                X=aimg,
                img_shape=(30, 30),
                tile_shape=(1, n_chains),
                tile_spacing=(1, 1)
            )
    
        # construct image
        image = Image.fromarray(image_data)
        image.save('samples.png')

        os.chdir('../')
        
        return


    def test_DBN(self, nhidden=1024, filtsize=12, finetune_lr=0.1, pretraining_epochs=100,
             pretrain_lr=0.01, k=1, training_epochs=1000,
             batch_size=10, plots_folder='dbn_plots'):
        """
        Demonstrates how to train and test a Deep Belief Network.
    
        This is demonstrated on MNIST.
    
        :type finetune_lr: float
        :param finetune_lr: learning rate used in the finetune stage
        :type pretraining_epochs: int
        :param pretraining_epochs: number of epoch to do pretraining
        :type pretrain_lr: float
        :param pretrain_lr: learning rate to be used during pre-training
        
        :type k: int
        :param k: number of Gibbs steps in CD/PCD
        :type training_epochs: int
        :param training_epochs: maximal number of iterations ot run the optimizer
        :type batch_size: int
        :param batch_size: the size of a minibatch
        """
    
        traindata_path='Z://Cristina//Section3//DeepLearning//allLpatches.pklz'
        labeldata_path='Z://Cristina//Section3//DeepLearning//allLabels.pklz'
        
        #############
        ## LOAD datasets
        #############
        datasets = self.load_data(traindata_path, labeldata_path)
    
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
    
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    
        # numpy random generator
        numpy_rng = np.random.RandomState(123)
        
        print '... building the model'
        # construct the Deep Belief Network
        dbn = DBN(numpy_rng=numpy_rng, 
                  n_ins=4*30*30,
                  hidden_layers_sizes=[nhidden, nhidden, nhidden],
                  n_outs=6)
    
        #########################
        # PRETRAINING THE MODEL #
        #########################
        if not os.path.isdir(plots_folder):
            os.makedirs(plots_folder)
        os.chdir(plots_folder)
        
        print '... getting the pretraining functions'
        pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                    batch_size=batch_size,
                                                    k=k)
    
        print '... pre-training the model'
        start_time = timeit.default_timer()
        
        ## Pre-train layer-wise
        for i in range(dbn.n_layers):
            # go through pretraining epochs
            for epoch in range(pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in range(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index,
                                                lr=pretrain_lr))
                                                
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print numpy.mean(c)
                
                # Plot filters after each dbn.n_layers                
                # Construct image from the weight matrix
                Xtmp = dbn.rbm_layers[i].W.get_value(borrow=True).T
                if(i==0):
                    imgX = Xtmp.reshape( Xtmp.shape[0], 4, 30, 30)
                    imgX0 = imgX[:,0,:,:]
                    image = Image.fromarray(
                        tile_raster_images(X=imgX0 , img_shape=(30, 30), tile_shape=(10, 10),
                                       tile_spacing=(1, 1)))
                if(i>0):
                    imgX = Xtmp.reshape( Xtmp.shape[0], 4, filtsize, filtsize)
                    imgX0 = imgX[:,0,:,:]
                    image = Image.fromarray(
                        tile_raster_images(X=imgX0 , img_shape=(filtsize, filtsize), tile_shape=(10, 10),
                                       tile_spacing=(1, 1)))
                
                                      
                image.save('filters_layer%i_epoch%i.png' % (i, epoch))
#                # prepare display    
#                fig, ax = plt.subplots()  
#                ax.imshow(image,  cmap="Greys_r")
#                ax.axes.get_xaxis().set_visible(False)
#                ax.axes.get_yaxis().set_visible(False)
    
        end_time = timeit.default_timer()

        print('The pretraining code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
                              
                              
        ########################
        # FINETUNING THE MODEL #
        ########################
        # get the training, validation and testing function for the model
        print '... getting the finetuning functions'
        train_fn, validate_model, test_model = dbn.build_finetune_functions(
            datasets=datasets,
            batch_size=batch_size,
            learning_rate=finetune_lr
        )
    
        print '... finetuning the model'
        
        # early-stopping parameters
        patience = 4 * n_train_batches  # look as this many examples regardless
        patience_increase = 2.    # wait this much longer when a new best is
                                  # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatches before checking the network
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
                    this_validation_loss = numpy.mean(validation_losses)
                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%'
                        % (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )
    
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
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))
    
                if patience <= iter:
                    done_looping = True
                    break
    
        end_time = timeit.default_timer()
        print(
            (
                'Optimization complete with best validation score of %f %%, '
                'obtained at iteration %i, '
                'with test performance %f %%'
            ) % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
        )
        print('The fine tuning code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time)/ 60.))
  
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
import matplotlib as mpl
import seaborn as sns
sns.set(color_codes=True)

from LogisticRegression import *
from MultilayerPerceptron import *
from rbm import *
from DBN import *
import utils

from utils import tile_raster_images
try:
    import PIL.Image as Image
except ImportError:
    import Image


class DLRBM_2D_temporal(object):
    """DLFuncs_2D_temporal Class 
    from DLFuncs_2D_temporal import *
    funcs = DLFuncs_2D_temporal()
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
 
    
    def load_wUdata(self, traindata_path, labeldata_path, trainUdata_path):
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
        
        lentrain = int(round(len(randuids)*0.8))
        lenvalid = int(round(len(randuids)*0.1))
        lentest = int(round(len(randuids)*0.1))
        
        # split ids based on random choice in 60% train, 20% valid and 20% test
        idstrain = [randuids[i] for i in range(lentrain)]
        idsvalid = [randuids[i] for i in range(lentrain,(lentrain+lenvalid))]
        idstest = [randuids[i] for i in range(lentrain+lenvalid,len(randuids))]
        
        
        randuidsU = self.randomList(list(range(len(Udatasets))))
        lentrainU = int(round(len(randuidsU)*0.8))
        lenvalidU = int(round(len(randuidsU)*0.1))
        lentestU = int(round(len(randuidsU)*0.1))
        
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
    
        rval = [(train_set_x, train_set_y), 
                (valid_set_x, valid_set_y),
                (test_set_x, test_set_y),
                (traindata, trainlabel),
                (validdata, validlabel),
                (testdata, testlabel)]
            
        return rval
        

        

    def test_RBM_timep(self, learning_rate=0.1, training_epochs=10,
                 batch_size=50,
                 n_chains=20, n_samples=5, output_folder='rbm_plots',
                 n_hidden=500):
        """
        Demonstrate how to train and afterwards sample from it using Theano.
        
        :param learning_rate: learning rate used for training the RBM
    
        :param training_epochs: number of epochs used for training
    
        :param dataset: path the the pickled dataset
    
        :param batch_size: size of a batch used to train the RBM
    
        :param n_chains: number of parallel Gibbs chains to be used for sampling
    
        :param n_samples: number of samples to plot for each chain
    
        from DLRBM_2D_temporal import *
        funcs = DLRBM_2D_temporal()
        funcs.test_RBM_timep()
    
        """
    
        traindata_path= 'allLpatches_10x10x5.pklz' #'allLpatches_subs_smaller.pklz' #'allLpatches.pklz'
        trainUdata_path= 'allUpatches_10x10x5.pklz'#'allUpatches_subs_smaller.pklz' #'allUpatches.pklz'
        labeldata_path= 'allLabels_10x10x5.pklz' #'allLabels_subs_smaller.pklz' #'allLabels.pklz'
        
        #############
        ## LOAD datasets
        #############
        datasets = self.load_wUdata(traindata_path, labeldata_path, trainUdata_path)
        
        train_set_x, train_set_y = datasets[0]
        np_train_x, np_train_y = datasets[3]
        valid_set_x, valid_set_y = datasets[1]
        np_valid_x, np_valid_y = datasets[4]        
        test_set_x, test_set_y = datasets[2]
        np_test_x, np_test_y = datasets[5]
    
        # inpect one image class 1
        Vol = np_train_x[0].reshape(5,10,10)
        imgslicestime = [Vol[0,:,:], Vol[1,:,:], Vol[2,:,:], Vol[3,:,:], Vol[4,:,:]]
        subslicestime = []  
        
        # Display image
        fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(16, 8))
        for k in range(1,5):
            ax[k-1,0].imshow(imgslicestime[k], cmap=plt.cm.gray)
            ax[k-1,0].set_axis_off()
            ax[k-1,0].set_adjustable('box-forced')
        
            # Display Original histogram
            ax[k-1,1].hist(imgslicestime[k].ravel(), bins=50, color='black')
            ax[k-1,1].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
            ax[k-1,1].set_xlabel('original')
            
            # substract volume
            subVol =  np.asarray(imgslicestime[k]) - np.asarray(imgslicestime[0])
            subslicestime.append(subVol)
        
            # Display normalized histogram
            ax[k-1,2].hist(subVol.ravel(), bins=50, color='black')
            ax[k-1,2].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
            ax[k-1,2].set_xlabel('substracted histogram')
       
            # display normalized 0-1
            ax[k-1,3].imshow(subVol, cmap=plt.cm.gray)
            ax[k-1,3].set_axis_off()
            ax[k-1,3].set_adjustable('box-forced')
        
        plt.show(block=False)
        
        #################
        # Substract pathces from pre-contrast
        #################
        subsnp_train_x = []
        subsnp_valid_x = []
        subsnp_test_x = []

        for img in np_train_x:
            Vol = img.reshape(5,10,10)
            imgslicestime = [Vol[0,:,:], Vol[1,:,:], Vol[2,:,:], Vol[3,:,:], Vol[4,:,:]]
            subslicestime = []  
            for k in range(1,5):
                # substract volume
                subVol =  np.asarray(imgslicestime[k]) - np.asarray(imgslicestime[0])
                subslicestime.append(subVol)
            #append
            subsnp_train_x.append( np.asarray(subslicestime).reshape(4*10*10) )          
        for img in np_valid_x:
            Vol = img.reshape(5,10,10)
            imgslicestime = [Vol[0,:,:], Vol[1,:,:], Vol[2,:,:], Vol[3,:,:], Vol[4,:,:]]
            subslicestime = []  
            for k in range(1,5):
                # substract volume
                subVol =  np.asarray(imgslicestime[k]) - np.asarray(imgslicestime[0])
                subslicestime.append(subVol)
            #append
            subsnp_valid_x.append( np.asarray(subslicestime).reshape(4*10*10) )
        for img in np_test_x:
            Vol = img.reshape(5,10,10)
            imgslicestime = [Vol[0,:,:], Vol[1,:,:], Vol[2,:,:], Vol[3,:,:], Vol[4,:,:]]
            subslicestime = []  
            for k in range(1,5):
                # substract volume
                subVol =  np.asarray(imgslicestime[k]) - np.asarray(imgslicestime[0])
                subslicestime.append(subVol)
            #append
            subsnp_test_x.append( np.asarray(subslicestime).reshape(4*10*10) )
        
#        train_set_x, train_set_y = self.shared_dataset(subsnp_train_x, np_train_y)
#        valid_set_x, valid_set_y = self.shared_dataset(subsnp_valid_x, np_valid_y)
#        test_set_x, test_set_y = self.shared_dataset(subsnp_test_x, np_test_y)
#        
#        subsdatasets = [(train_set_x, train_set_y), 
#                        (valid_set_x, valid_set_y), 
#                        (test_set_x, test_set_y) ]
                        
            
        print "n_chains %d " % n_chains
        print "n_hidden %d " %  n_hidden
        print "training_epochs %d " %  training_epochs
        print "batch_size %d " %  batch_size
        print "learning_rate %f " %  learning_rate      
        print "n_samples %d " %  n_samples
        print "output_folder %s " %  output_folder
               
        ##############################
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    
        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch
        x = T.matrix('x')  # the data is presented as rasterized images
    
        rng = numpy.random.RandomState(123)
        theano_rng = RandomStreams(rng.randint(2 ** 30))
    
        # initialize storage for the persistent chain (state = hidden
        # layer of chain)
        persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                     dtype=theano.config.floatX),
                                         borrow=True)
    
        # construct the RBM class
        rbm = RBM(input=x, n_visible=5*10*10,
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
    
            print('Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost))
    
            # Plot filters after each training epoch
            plotting_start = timeit.default_timer()
            # Construct image from the weight matrix
            Wrbm = rbm.W.get_value(borrow=True).T
            
            fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(16, 6))
            for k in range(1,5):
                image = utils.tile_images(
                        X=Wrbm.reshape( Wrbm.shape[0], 5, 10, 10)[:,k,:,:],
                        img_shape=(10,10),
                        tile_shape=(25,20),
                        tile_spacing=(0, 0) )
                
                im = axes[k-1].imshow(image, vmin=np.min(image.ravel()), vmax=np.max(image.ravel()), interpolation='nearest', cmap="Greys_r")  
                axes[k-1].get_xaxis().set_visible(False)
                axes[k-1].get_yaxis().set_visible(False)
                
            cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
            plt.colorbar(im, cax=cax, **kw)  
            plt.show(block=False)
            fig.savefig('filters_at_epoch_%i.png' % epoch)
            
            plotting_stop = timeit.default_timer()
            plotting_time += (plotting_stop - plotting_start)
    
        end_time = timeit.default_timer()
        pretraining_time = (end_time - start_time) - plotting_time
        print ('Plotting took %f minutes' % (plotting_time / 60.))
        print ('Training took %f minutes' % (pretraining_time / 60.))

        #####################################
        # save the best model
        #####################################
        with open('bestRBM_10x10x5.obj', 'wb') as fp:
            pickle.dump(rbm, fp)
        
        #####################################
        # Open best model
        #####################################
        with open('bestRBM_10x10x5.obj', 'rb') as fp:
            rbm = pickle.load(fp)
            
        #################################
        #     Sampling from the RBM     #
        #################################
        # find out the number of test samples
        number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]
    
        # pick random test examples, with which to initialize the persistent chain
        test_idx = rng.randint(number_of_test_samples - n_chains)
        persistent_vis_chain = theano.shared(
            numpy.asarray(
                test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
                dtype=theano.config.floatX
            )
        )
        plot_every = 1000
        # define one step of Gibbs sampling (mf = mean-field) define a
        # function that does `plot_every` steps before returning the
        # sample for plotting
        (
            [
                presig_hids,
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
        sub_pre = numpy.zeros(
            (11 * n_samples + 1, 11 * n_chains - 1),
            dtype='uint8')
        sub_post1 = numpy.zeros(
            (11 * n_samples + 1, 11 * n_chains - 1),
            dtype='uint8')
        sub_post2 = numpy.zeros(
            (11 * n_samples + 1, 11 * n_chains - 1),
            dtype='uint8')
        sub_post3 = numpy.zeros(
            (11 * n_samples + 1, 11 * n_chains - 1),
            dtype='uint8')
        sub_post4 = numpy.zeros(
            (11 * n_samples + 1, 11 * n_chains - 1),
            dtype='uint8')
        
        fig, axes = plt.subplots(ncols=1, nrows=5)
        for idx in range(n_samples):
            # generate `plot_every` intermediate samples that we discard,
            # because successive samples in the chain are too correlated
            vis_mf, vis_sample = sample_fn()
            print(' ... plotting sample %d' % idx)
            Xvis = vis_mf.reshape( vis_mf.shape[0],5,100)
            
            sub_pre[11 * idx:11 * idx + 10, :] = utils.tile_raster_images(
                X=Xvis[:,0,:],
                img_shape=(10, 10),
                tile_shape=(1, n_chains),
                tile_spacing=(1, 1)
            )
            sub_post1[11 * idx:11 * idx + 10, :] = utils.tile_raster_images(
                X=Xvis[:,1,:],
                img_shape=(10, 10),
                tile_shape=(1, n_chains),
                tile_spacing=(1, 1)
            )
            sub_post2[11 * idx:11 * idx + 10, :] = utils.tile_raster_images(
                X=Xvis[:,2,:],
                img_shape=(10, 10),
                tile_shape=(1, n_chains),
                tile_spacing=(1, 1)
            )
            sub_post3[11 * idx:11 * idx + 10, :] = utils.tile_raster_images(
                X=Xvis[:,3,:],
                img_shape=(10, 10),
                tile_shape=(1, n_chains),
                tile_spacing=(1, 1)
            )
            sub_post4[11 * idx:11 * idx + 10, :] = utils.tile_raster_images(
                X=Xvis[:,4,:],
                img_shape=(10, 10),
                tile_shape=(1, n_chains),
                tile_spacing=(1, 1)
            )
    
        # construct image
        axes[0].imshow( Image.fromarray( sub_pre), cmap="Greys_r")
        axes[1].imshow( Image.fromarray( sub_post1), cmap="Greys_r")
        axes[2].imshow( Image.fromarray( sub_post2), cmap="Greys_r")
        axes[3].imshow( Image.fromarray( sub_post3), cmap="Greys_r") 
        axes[4].imshow( Image.fromarray( sub_post4), cmap="Greys_r")  
        axes[0].get_xaxis().set_visible(False)
        axes[0].get_yaxis().set_visible(False)
        axes[1].get_xaxis().set_visible(False)
        axes[1].get_yaxis().set_visible(False)
        axes[2].get_xaxis().set_visible(False)
        axes[2].get_yaxis().set_visible(False)
        axes[3].get_xaxis().set_visible(False)
        axes[3].get_yaxis().set_visible(False)
        axes[4].get_xaxis().set_visible(False)
        axes[4].get_yaxis().set_visible(False)
        plt.show() 
        item='test'
        #show and save                     
        fig.savefig('samples_fromchain'+str(item)+'.pdf')

        return 
    
    
    def test_DBN(self, finetune_lr=0.1, pretraining_epochs=10,
             pretrain_lr=0.01, k=1, training_epochs=10, batch_size=10,
             hidden_layers_sizes=[400,400], sidelen=20,
             output_folder='DBN_plots', item='test'):
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
        :type dataset: string
        :param dataset: path the the pickled dataset
        :type batch_size: int
        :param batch_size: the size of a minibatch
        
        
        from DLRBM_2D_temporal import *
        funcs = DLRBM_2D_temporal()
        funcs.test_DBN()
        
       
        DBN.test_DBN()
    
        """
    
        traindata_path= 'allLpatches_10x10x5.pklz' #'allLpatches_subs_smaller.pklz' #'allLpatches.pklz'
        trainUdata_path= 'allUpatches_10x10x5.pklz'#'allUpatches_subs_smaller.pklz' #'allUpatches.pklz'
        labeldata_path= 'allLabels_10x10x5.pklz' #'allLabels_subs_smaller.pklz' #'allLabels.pklz'
        
        #############
        ## LOAD datasets
        #############
        datasets = self.load_wUdata(traindata_path, labeldata_path, trainUdata_path)
    
        train_set_x, train_set_y = datasets[0]
        np_train_x, np_train_y = datasets[3]
        valid_set_x, valid_set_y = datasets[1]
        np_valid_x, np_valid_y = datasets[4]        
        test_set_x, test_set_y = datasets[2]
        np_test_x, np_test_y = datasets[5]
        

        #########################
        # FORMAT THE DATA
        #########################
        ## transform to pixel intensities between 0 and 1
        tnp_train_x= list( utils.scale_to_unit_interval( np.asarray(np_train_x) ) )
        tnp_valid_x = list( utils.scale_to_unit_interval( np.asarray(np_valid_x) ) )
        tnp_test_x = list( utils.scale_to_unit_interval( np.asarray(np_test_x) ) )
        
        #################
        # Substract pathces from pre-contrast
        #################
#        subsnp_train_x = []
#        subsnp_valid_x = []
#        subsnp_test_x = []
#
#        for img in np_train_x:
#            Vol = img.reshape(5,10,10)
#            imgslicestime = [Vol[0,:,:], Vol[1,:,:], Vol[2,:,:], Vol[3,:,:], Vol[4,:,:]]
#            subslicestime = []  
#            for k in range(1,5):
#                # substract volume
#                subVol =  np.asarray(imgslicestime[k]) - np.asarray(imgslicestime[0])
#                subslicestime.append(subVol)
#            #append
#            subsnp_train_x.append( np.asarray(subslicestime).reshape(4*10*10) )          
#        for img in np_valid_x:
#            Vol = img.reshape(5,10,10)
#            imgslicestime = [Vol[0,:,:], Vol[1,:,:], Vol[2,:,:], Vol[3,:,:], Vol[4,:,:]]
#            subslicestime = []  
#            for k in range(1,5):
#                # substract volume
#                subVol =  np.asarray(imgslicestime[k]) - np.asarray(imgslicestime[0])
#                subslicestime.append(subVol)
#            #append
#            subsnp_valid_x.append( np.asarray(subslicestime).reshape(4*10*10) )
#        for img in np_test_x:
#            Vol = img.reshape(5,10,10)
#            imgslicestime = [Vol[0,:,:], Vol[1,:,:], Vol[2,:,:], Vol[3,:,:], Vol[4,:,:]]
#            subslicestime = []  
#            for k in range(1,5):
#                # substract volume
#                subVol =  np.asarray(imgslicestime[k]) - np.asarray(imgslicestime[0])
#                subslicestime.append(subVol)
#            #append
#            subsnp_test_x.append( np.asarray(subslicestime).reshape(4*10*10) )
#    
#    
#    
#        #########################
#        # FORMAT THE DATA
#        #########################
#        ## transform to pixel intensities between 0 and 1
#        tsubsnp_train_x = list( utils.scale_to_unit_interval( np.asarray(subsnp_train_x) ) )
#        tsubsnp_valid_x = list( utils.scale_to_unit_interval( np.asarray(subsnp_valid_x) ) )
#        tsubsnp_test_x = list( utils.scale_to_unit_interval( np.asarray(subsnp_test_x) ) )
#        
#        # inpect one image class 1/0
#        tVol = tsubsnp_train_x[29594].reshape(4,10,10)
#        timgslicestime = [tVol[0,:,:], tVol[1,:,:], tVol[2,:,:], tVol[3,:,:]]
#        
#        # inpect one image class 1/0
#        Vol = np_train_x[29594].reshape(5,10,10)
#        imgslicestime = [Vol[0,:,:], Vol[1,:,:], Vol[2,:,:], Vol[3,:,:], Vol[4,:,:]]
#        
#        # Display image
#        fig, ax = plt.subplots(nrows=4, ncols=6, figsize=(16, 8))
#        for k in range(1,5):
#            ax[k-1,0].imshow(imgslicestime[k], cmap=plt.cm.gray)
#            ax[k-1,0].set_axis_off()
#            ax[k-1,0].set_adjustable('box-forced')
#        
#            # Display Original histogram
#            ax[k-1,1].hist(imgslicestime[k].ravel(), bins=50, color='black')
#            ax[k-1,1].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
#            ax[k-1,1].set_xlabel('original')
#            
#            # substract volume
#            subVol =  np.asarray(imgslicestime[k]) - np.asarray(imgslicestime[0])
#            subslicestime.append(subVol)
#        
#            # Display subtracted histogram
#            ax[k-1,2].hist(subVol.ravel(), bins=50, color='black')
#            ax[k-1,2].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
#            ax[k-1,2].set_xlabel('substracted histogram')
#       
#            # display  subtracted  
#            ax[k-1,3].imshow(subVol, cmap=plt.cm.gray)
#            ax[k-1,3].set_axis_off()
#            ax[k-1,3].set_adjustable('box-forced')
#            
#            # display  pixels 0-1 subtracted  
#            ax[k-1,4].imshow(timgslicestime[k-1], cmap=plt.cm.gray)
#            ax[k-1,4].set_axis_off()
#            ax[k-1,4].set_adjustable('box-forced')
#            
#            # display pixels 0-1 subtracted histogram
#            ax[k-1,5].hist(timgslicestime[k-1].ravel(), bins=50, color='black')
#            ax[k-1,5].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
#            ax[k-1,5].set_xlabel(' pixels 0-1 subtracted histogram')
#        
#        plt.show(block=False)
        
        train_set_x, train_set_y = self.shared_dataset( tnp_train_x, np_train_y )
        valid_set_x, valid_set_y = self.shared_dataset( tnp_valid_x, np_valid_y )
        test_set_x, test_set_y = self.shared_dataset( tnp_test_x, np_test_y )
        
        datasets = [    (train_set_x, train_set_y), 
                        (valid_set_x, valid_set_y), 
                        (test_set_x, test_set_y)    ]
        
    
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        
        # numpy random generator
        numpy_rng = numpy.random.RandomState(123)
        print '... building the model'
        
        # construct the Deep Belief Network
        dbn = DBN(numpy_rng=numpy_rng, n_ins=5*10*10,
                  hidden_layers_sizes=hidden_layers_sizes,
                  n_outs=2)
    
        #########################
        # PRETRAINING THE MODEL #
        #########################
        DBN_avg_costs = []
        DBN_iter = []
        layer_i = []
        
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
                if( epoch % 50 == 0):
                    pretrain_lr = pretrain_lr/10
                    
                for batch_index in range(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index,
                                                lr=pretrain_lr))
                 # append      
                DBN_avg_costs.append(  np.mean(c) )
                DBN_iter.append(epoch)
                layer_i.append(i)
                
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print numpy.mean(c)
    
        end_time = timeit.default_timer()
    
        print 'The pretraining code ran for %.2fm' % ((end_time - start_time) / 60.)
        
        #####################################
        # Plot images in 2D
        #####################################   
        Wrbm = dbn.rbm_layers[0].W.get_value(borrow=True).T
        
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(16, 6))
        axes = axes.flatten()
        for k in range(1,5):
            image = Image.fromarray(
                    utils.tile_raster_images(
                    X=Wrbm.reshape( Wrbm.shape[0], 5, 10, 10)[:,k,:,:],
                    img_shape=(10,10),
                    tile_shape=(sidelen,sidelen),
                    tile_spacing=(1, 1) ))
            
            im = axes[k-1].imshow(image, cmap="Greys_r")  
            axes[k-1].get_xaxis().set_visible(False)
            axes[k-1].get_yaxis().set_visible(False)
            
        cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
        plt.colorbar(im, cax=cax, **kw)  
        fig.savefig(output_folder+'/filters_dbn'+str(item)+'.pdf')
                               
        ##############
        # Format      
        #################           
        LLdata = [float(L) for L in DBN_avg_costs]
        LLiter = [float(it) for it in DBN_iter]
        LLilayer = [ilayer for ilayer in layer_i]
        dfpredata = pd.DataFrame( LLdata )
        dfpredata.columns = ['LL_iter']
        dfpredata['iter'] = LLiter
        dfpredata['layer'] = LLilayer
        
                           
        ########################
        # FINETUNING THE MODEL #
        ########################
        # get the training, validation and testing function for the model
        print '... getting the finetuning functions'
        train_fn, validate_model, test_model = dbn.build_finetune_functions(
            datasets=datasets,
            batch_size=batch_size,
            learning_rate=finetune_lr)
    
        ############
        ### for plotting likelihood or cost, accumulate returns of train_model
        ############
        minibatch_avg_costs = []
        minibatch_iter = []
        minibatch_loss = []
        
        print '... finetuning the model'
        # early-stopping parameters
        patience = 1000 * n_train_batches  # look as this many examples regardless
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
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))
    
                if patience <= iter:
                    done_looping = True
                    break
    
        end_time = timeit.default_timer()
        
        print('Optimization complete with best validation score of %f %%, '
                'obtained at iteration %i, '
                'with test performance %f %%') % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
                
        print 'The fine tuning code ran for %.2fm' % ((end_time - start_time)/ 60.)
    
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
    
        ###############
        ## Predictions
        ###############
        # get training data in numpy format   
        X,y = tnp_train_x, np_train_y # datasets[3]  
        Xtrain = np.asarray(X)
        ytrain = utils.makeMultiClass(y)
        # get valid data in numpy format   
        X,y = tnp_valid_x, np_valid_y # datasets[4]
        Xvalid = np.asarray(X)
        yvalid = utils.makeMultiClass(y)
        # get test data in numpy format           
        X,y = tnp_test_x, np_test_y  # datasets[5]  
        Xtest = np.asarray(X)
        ytest = utils.makeMultiClass(y)
                 
        
        ###############
        # predicting using the SDA
        ###############
        # in train
        predtrain = dbn.predict_functions(Xtrain).argmax(1)
        # let's see how the network did
        y = ytrain.argmax(1)
        e0 = 0.0; y0 = len([0 for yi in range(len(y)) if y[yi]==0])
        e1 = 0.0; y1 = len([1 for yi in range(len(y)) if y[yi]==1])
        for i in range(len(y)):
            if(y[i] == 1):
                e1 += y[i]==predtrain[i]
            if(y[i] == 0):
                e0 += y[i]==predtrain[i]
    
        # printing the result, this structure should result in 80% accuracy
        Acutrain0=100*e0/y0
        Acutrain1=100*e1/y1
        print "Train Accuracy for class 0: %2.2f%%"%(Acutrain0)
        print "Train Accuracy for class 1: %2.2f%%"%(Acutrain1)  
        
        # in Valid
        predvalid = dbn.predict_functions(Xvalid).argmax(1)
        # let's see how the network did
        y = yvalid.argmax(1)
        e0 = 0.0; y0 = len([0 for yi in range(len(y)) if y[yi]==0])
        e1 = 0.0; y1 = len([1 for yi in range(len(y)) if y[yi]==1])
        for i in range(len(y)):
            if(y[i] == 1):
                e1 += y[i]==predvalid[i]
            if(y[i] == 0):
                e0 += y[i]==predvalid[i]
    
        # printing the result, this structure should result in 80% accuracy
        Acuvalid0=100*e0/y0
        Acuvalid1=100*e1/y1
        print "Valid Accuracy for class 0: %2.2f%%"%(Acuvalid0)
        print "Valid Accuracy for class 1: %2.2f%%"%(Acuvalid1) 
        
        # in Xtest
        predtest = dbn.predict_functions(Xtest).argmax(1)
        # let's see how the network did
        y = ytest.argmax(1)
        e0 = 0.0; y0 = len([0 for yi in range(len(y)) if y[yi]==0])
        e1 = 0.0; y1 = len([1 for yi in range(len(y)) if y[yi]==1])
        for i in range(len(y)):
            if(y[i] == 1):
                e1 += y[i]==predtest[i]
            if(y[i] == 0):
                e0 += y[i]==predtest[i]
    
        # printing the result, this structure should result in 80% accuracy
        Acutest0=100*e0/y0
        Acutest1=100*e1/y1
        print "Test Accuracy for class 0: %2.2f%%"%(Acutest0)
        print "Test Accuracy for class 1: %2.2f%%"%(Acutest1) 
            
    
        return [dfpredata, dfinedata, dbn, 
                Acutrain0, Acutrain1,
                Acuvalid0, Acuvalid1,
                Acutest0, Acutest1]
 
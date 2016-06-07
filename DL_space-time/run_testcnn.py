# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:42:56 2016

@author: windows
"""

from DeepLearnFuncs import *
import six.moves.cPickle as cPickle
import theano 
from theano import tensor as T 
from theano.tensor.nnet import conv
import numpy as np

def run_ConvolutionalLenet5():
    # start by importing Deep Learning Funcs
    funcs = DeepLearnFuncs()
    
    learning_rate=0.1
    n_epochs=1000
    nkerns = [20,50]
    batch_size = 500
    
    # test convolutional operator
    rng = np.random.RandomState(23455)
    
    # instantiate 4D tensor for input 
    input = T.tensor4(name='input')
    
    # **4** features maps (4 post-contrast images) of size 30x30. 
    # We use **3** convolutional filters with 3x3 receptive fields
    # initialize shared variable for weights. 
    w_shp = (2, 4, 3, 3)
    w_bound = np.sqrt(4 * 3 * 3) 
    W = theano.shared( np.asarray( rng.uniform( 
            low=-1.0 / w_bound, 
            high=1.0 / w_bound, 
            size=w_shp), dtype=input.dtype), name ='W')
    
    # initialize shared variable for bias (1D tensor) with random values 
    # IMPORTANT: biases are usually initialized to zero. 
    # Here we simply apply the convolutional layer to 
    # an image without learning the parameters. 
    # We therefore initialize # them to random values to "simulate" learning. 
    b_shp = (2,) 
    b = theano.shared(np.asarray( rng.uniform(
            low=-.5, 
            high=.5, 
            size=b_shp), dtype=input.dtype), name ='b')
    
    # build symbolic expression that computes the convolution of input with filters in w 
    conv_out = conv.conv2d(input, W)

    # build symbolic expression to add bias and apply activation function, i.e. produce neural 
    # A few words on ‘‘dimshuffle‘‘ : 
    # ‘‘dimshuffle‘‘ is a powerful tool in reshaping a tensor; 
    # what it allows you to do is to shuffle dimension around # but also to insert new ones along which the tensor will be 
    # broadcastable; # dimshuffle(’x’, 2, ’x’, 0, 1) # This will work on 3d tensors with no broadcastable 
    # dimensions. The first dimension will be broadcastable, 
    # then we will have the third dimension of the input tensor as # the second of the resulting tensor, etc. If the tensor has 
    # shape (20, 30, 40), the resulting tensor will have dimensions 
    # (1, 40, 1, 20, 30). (AxBxC tensor is mapped to 1xCx1xAxB tensor) 
    # More examples: # dimshuffle(’x’) -> make a 0d (scalar) into a 1d vector 
    # dimshuffle(0, 1) -> identity # dimshuffle(1, 0) -> inverts the first and second dimensions 
    # dimshuffle(’x’, 0) -> make a row out of a 1d vector (N to 1xN) 
    # dimshuffle(0, ’x’) -> make a column out of a 1d vector (N to Nx1) 
    # dimshuffle(2, 0, 1) -> AxBxC to CxAxB # dimshuffle(0, ’x’, 1) -> AxB to Ax1xB 
    # dimshuffle(1, ’x’, 0) -> AxB to Bx1xA 
    output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

    # create theano function to compute filtered images 
    fconvol = theano.function([input], output)

    # dimensions are (height, width, channel) 
    # We can test it on some examples from test test 
    test_labels = cPickle.load(open('test_labels.pkl'))
    test_set = cPickle.load(open('test_set.pkl'))
    test_set_x, test_set_y = test_set 
    test_set_x = test_set_x.get_value() 
    
    # We can test it on some examples from test test   
    print ("values for the first 20 examples in test set:") 
    print test_labels[0:19]

    # put image in 4D tensor of shape (1, 4, height, width) 
    # grab test set image, reshape to proper order and transpose
    img = np.reshape(test_set_x[0],(4,30,30))    
    #img = np.asarray(img, dtype='float64') 
    img_ = img.reshape(1, 4, 30, 30) 
    filtered_img = fconvol(img_)
    
    # prepare display    
    fig, ax = plt.subplots(ncols=2, nrows=4)
    
    # show original imgs
    ax[0,0].imshow(img[0,:,:], cmap="Greys_r")
    ax[1,0].imshow(img[1,:,:], cmap="Greys_r")
    ax[2,0].imshow(img[2,:,:], cmap="Greys_r")
    ax[3,0].imshow(img[3,:,:], cmap="Greys_r")
    # show convolved filters
    ax[1,1].imshow(filtered_img[0,0,:,:], cmap="Greys_r")
    ax[2,1].imshow(filtered_img[0,1,:,:], cmap="Greys_r")
    plt.show() 


    ############
    # train lenet                 
    ############
    dfLLdata = funcs.evaluate_lenet5(learning_rate, n_epochs, nkerns, batch_size)
                 
    ############
    ### plotting likelihood or cost
    ### the cost we minimize during training is the negative log likelihood of
    ############
    x = dfLLdata['iter'].values
    y = dfLLdata['LL_iter'].values
    plt.figure()
    plt.plot(x, y, 'bo--')
    plt.xlabel('iterations', fontsize=14)
    plt.ylabel('negative log likelihood', fontsize=14)
    plt.title('MLP: learning_rate = '+str(learning_rate)+' batch_size = '+str(batch_size), fontsize=14)

   
    ############
    ### plotting likelihood or cost
    ############     
    x = dfLLdata['iter'].values
    y = dfLLdata['0-1-loss'].values
    plt.figure()
    plt.plot(x, y, 'bo--')
    plt.xlabel('iterations')
    plt.ylabel('0-1-loss %')
    plt.title('MLP: learning_rate = '+str(learning_rate)+' batch_size = '+str(batch_size))

    


if __name__ == '__main__':
    # run testing Convolutional Lenet5
    run_ConvolutionalLenet5()
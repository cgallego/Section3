# -*- coding: utf-8 -*-
"""
Created on Mon May 30 16:13:35 2016

@author: windows
"""

from DeepLearnFuncs import *
import six.moves.cPickle as cPickle

def run_MultilayerPerceptron():
    # start by importing Deep Learning Funcs
    funcs = DeepLearnFuncs()
    
    learning_rate=0.095
    n_epochs=100
    batch_size=300
    L1_reg=0.00
    L2_reg=0.01
    n_hidden=1500
                 
    dfLLdata = funcs.test_mlp(learning_rate, L1_reg, L2_reg, n_epochs,
                 batch_size, n_hidden)
                 
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

     ############
    # load the saved model 
    ############
    # We can test it on some examples from test test 
    test_labels = cPickle.load(open('test_labels.pkl'))  
    
    # We can test it on some examples from test test 
    test_set = cPickle.load(open('test_set.pkl'))
    test_set_x, test_set_y = test_set 
    test_set_x = test_set_x.get_value() 
    
    print ("Predicted/Labels values for the first 10 examples in test set:") 
    print test_labels[13:18]
    
    fig, ax = plt.subplots(ncols=1, nrows=1)
    img = np.reshape(test_set_x[13],(30,30))    
    ax.imshow(img, cmap="Greys_r")
    plt.show()
   
    
if __name__ == '__main__':
    # run testing MultilayerPerceptron
    run_MultilayerPerceptron()
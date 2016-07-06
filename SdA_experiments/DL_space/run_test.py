# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:37:58 2016

@author: windows
"""

from DeepLearnFuncs import *
import six.moves.cPickle as cPickle

def run_LogisticRegression():
    # start by importing Deep Learning Funcs
    funcs = DeepLearnFuncs()
    
    learning_rate=0.0001
    n_epochs=1000
    batch_size=2
    dfLLdata = funcs.sgd_optimization(learning_rate, n_epochs, batch_size)
 
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
    plt.title('LogReg: learning_rate = '+str(learning_rate)+' batch_size = '+str(batch_size), fontsize=14)

   
    ############
    ### plotting likelihood or cost
    ############     
    x = dfLLdata['iter'].values
    y = dfLLdata['0-1-loss'].values
    plt.figure()
    plt.plot(x, y, 'bo--')
    plt.xlabel('iterations')
    plt.ylabel('0-1-loss %')
    plt.title('LogReg: learning_rate = '+str(learning_rate)+' batch_size = '+str(batch_size))

    ############
    # load the saved model 
    ############
    classifier = cPickle.load(open('best_model.pkl'))
    
    # compile a predictor function 
    predict_model = theano.function( inputs=[classifier.input], outputs=classifier.y_pred)
    
    # We can test it on some examples from test test 
    test_set = cPickle.load(open('test_set.pkl'))
    test_set_x, test_set_y = test_set 
    test_set_x = test_set_x.get_value() 
    
    # We can test it on some examples from test test 
    test_labels = cPickle.load(open('test_labels.pkl'))
    
    predicted_values = predict_model(test_set_x[13:18]) 
    print ("Predicted/Labels values for the first 10 examples in test set:") 
    print predicted_values
    print test_labels[13:18]
    
    fig, ax = plt.subplots(ncols=1, nrows=1)
    img = np.reshape(test_set_x[13],(30,30))    
    ax.imshow(img, cmap="Greys_r")
    plt.show()
   

if __name__ == '__main__':
    
    # run testing LogisticRegression
    run_LogisticRegression()
    

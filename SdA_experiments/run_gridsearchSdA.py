 
from SDA_layers import StackedDA  
from DLFuncs_SdA import *
import utils
import utils2
import seaborn as sns
import pandas as pd
import numpy as np

import itertools
import six.moves.cPickle as pickle

    
def SdA_gridsearch():
    funcs = DLFuncs_SdA()
    traindata_path='Z://Cristina//Section3//SdA_experiments//allLpatches.pklz'
    trainUdata_path='Z://Cristina//Section3//SdA_experiments//allUpatches.pklz'
    labeldata_path='Z://Cristina//Section3//SdA_experiments//allLabels.pklz'
    
    datasets = funcs.load_wUdata(traindata_path, labeldata_path, trainUdata_path)
 
    ############
    ### plotting labels 
    ############
    dftrain = pd.DataFrame();   dfvalid = pd.DataFrame();   dftest = pd.DataFrame();   
    # get training data in numpy format   
    X,y = datasets[3]
    Xtrain = np.asarray(X)
    # extract one img
    Xtrain = Xtrain.reshape(Xtrain.shape[0], 4, 900)
    Xtrain = Xtrain[:,0,:]
    ytrain = utils2.makeMultiClass(y)
    dftrain['y'] = pd.Series(['y0' if y[yi]==0 else 'y1' for yi in range(len(ytrain))])
    dftrain['group'] = pd.Series(np.repeat('train',len(y)))
    
    # get valid data in numpy format   
    X,y = datasets[4]
    Xvalid = np.asarray(X)
    # extract one img
    Xvalid = Xvalid.reshape(Xvalid.shape[0], 4, 900)
    Xvalid = Xvalid[:,0,:]
    yvalid = utils2.makeMultiClass(y)
    dfvalid['y'] = pd.Series(['y0' if y[yi]==0 else 'y1' for yi in range(len(yvalid))])
    dfvalid['group'] = pd.Series(np.repeat('valid',len(y)))
    
    # get training data in numpy format   
    X,y = datasets[5]
    Xtest = np.asarray(X)
    # extract one img
    Xtest = Xtest.reshape(Xtest.shape[0], 4, 900)
    Xtest = Xtest[:,0,:]
    ytest = utils2.makeMultiClass(y)
    dftest['y'] = pd.Series(['y0' if y[yi]==0 else 'y1' for yi in range(len(ytest))])
    dftest['group'] = pd.Series(np.repeat('test',len(y)))
    
    # concat and plot
    df = pd.concat([dftrain, dfvalid, dftest], axis=0)
    sns.set(style="whitegrid")
    # Draw a nested barplot to show survival for class and sex
    g = sns.countplot(x="group", hue="y", palette="muted", data=df)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x(), height+ 3, '%1.2f'%(height))
    fig = g.get_figure()
    fig.savefig('grid_searchResults/datasets.png')

    ############
    ### Define grid search parameters
    ############
    nlayers = [1,2,3,4]
    nhiddens = [100,225,400,900]
    hidden_layers_sidelen = [10,15,20,30] 
    nnoise_rate = [0.35]
    nalpha = [0.01,0.1,0.5] 
    batch_size = [500,100,10]
    # note batch_size is fixed to 1 and epochs is fixed to 25
    k=0
    BestAveAccuracy = 0
    #dfresults = pd.DataFrame()# when first time
    
    ###########
    ## Process Resuts
    ###########
    pkl_filegridS = open('grid_searchResults/gridSearch_results2.pkl','rb')
    dfresults = pickle.load(pkl_filegridS)
    print dfresults
    #items = itertools.product(nlayers, nhiddens, nnoise_rate, nalpha, batch_size)
    #item=items.next()
    
    for item in itertools.product(nlayers, nhiddens, nnoise_rate, nalpha, batch_size): 
        k+=1
        print(k,item)
        
        if(k>90):
            # setup the training functions
            nlayer = item[0]
            nhidden = item[1]
            sidelen = hidden_layers_sidelen[ nhiddens.index(nhidden) ]
            noiseRate = item[2]
            alpha = item[3]
            batchS = item[4]
            if(nlayer == 1):
                StackedDA_layers = [nhidden]
            if(nlayer == 2):
                StackedDA_layers = [nhidden,nhidden]
            if(nlayer == 3):
                StackedDA_layers = [nhidden,nhidden,nhidden]
                
            # building the SDA
            sDA = StackedDA(StackedDA_layers, alpha, item)
            
            # pre-trainning the SDA
            sDA.pre_train(Xtrain, noise_rate=noiseRate, epochs=200, batchsize=batchS)
            
            #####################################
            # saving a PNG representation of the first layer
            #####################################
            # Plot images in 2D       
            W0 = sDA.Layers[0].W.T[:, 1:]
            imageW0 = Image.fromarray(
                utils.tile_raster_images(X=W0 , img_shape=(30, 30), 
                                   tile_shape=(10, 10),
                                   tile_spacing=(1, 1)))
        
            #show and save                     
            imageW0.save('grid_searchResults/filters_hidden_1stlayer_'+str(item)+'.png')
            # prepare display    
            fig, ax = plt.subplots()  
            ax.imshow(imageW0,  cmap="Greys_r")
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            
           # W = sDA.Layers[0].W.T[:, 1:]
            #utils2.saveTiles(W, img_shape= (30,30), tile_shape=(20,10), filename='grid_searchResults/'+'.png")
        
            # adding the final layer
            sDA.finalLayer(Xtrain, ytrain, epochs=20)
        
            # trainning the whole network
            sDA.fine_tune(Xtrain, ytrain, epochs=20)
        
            # predicting using the SDA
            pred = sDA.predict(Xvalid).argmax(1)
            
            # let's see how the network did
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
            print "accuracy for class 0: %2.2f%%"%(100*e0/y0)
            print "accuracy for class 1: %2.2f%%"%(100*e1/y1)
            
            # append results
            accuracy0 = 100*e0/y0
            accuracy1 = 100*e1/y1
            itemresults = item + (accuracy0, accuracy1)
            dSresultsiter =  pd.DataFrame(data=np.array(itemresults)).T
            dSresultsiter.columns=['nlayers', 'nhiddens', 'nnoise_rate', 'nalpha', 'batchsize', 'accuracy0','accuracy1']
                  
            dfresults = dfresults.append(dSresultsiter)      
            AveAccuracy = (accuracy0 + accuracy1)/2
            # find best model so far
            if( AveAccuracy > BestAveAccuracy):
                bestsDA = sDA
                BestAveAccuracy = AveAccuracy
                print("best Accuracy = %d, for SdA:" % AveAccuracy)
                print(bestsDA)
                
            # save the best model
            with open('grid_searchResults/gridSearch_results2.pkl', 'wb') as f:
                pickle.dump(dfresults, f)
            
        ### continue
        print(dfresults)

    return 
    
    
if __name__ == '__main__':
    SdA_gridsearch()
    
    
    ###########
    ## Process Resuts
    ###########
    pkl_filegridS = open('grid_searchResults/gridSearch_results2.pkl','rb')
    dfresults = pickle.load(pkl_filegridS)
    print dfresults
    
    
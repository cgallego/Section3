 
from SDA_layers import StackedDA  
from DLFuncs_SdA import *
import utils2
import seaborn as sns
import pandas as pd
import itertools
    
def demo():
    funcs = DLFuncs_SdA()
    traindata_path='Z://Cristina//Section3//SdA_experiments//allLpatches.pklz'
    trainUdata_path='Z://Cristina//Section3//SdA_experiments//allUpatches.pklz'
    labeldata_path='Z://Cristina//Section3//SdA_experiments//allLabels.pklz'
    
    datasets = funcs.load_wUdata(traindata_path, labeldata_path, trainUdata_path)
    
    dftrain = pd.DataFrame();   dfvalid = pd.DataFrame();   dftest = pd.DataFrame();   
 
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
    
    

a = ["1"]
b = ["0"]
c = ["a","b","c"]
d = ["d","e","f"]

for item in itertools.product(a, b, c, d): 
    print(item)
    

    
    # building the SDA
    sDA = StackedDA([400, 400])

    # pre-trainning the SDA
    sDA.pre_train(Xvalid, noise_rate=0.3, epochs=1)

    # saving a PNG representation of the first layer
    W = sDA.Layers[0].W.T[:, 1:]
    utils2.saveTiles(W, img_shape= (30,30), tile_shape=(20,10), filename="res_dA.png")

    # adding the final layer
    sDA.finalLayer(Xtrain, ytrain, epochs=1)

    # trainning the whole network
    sDA.fine_tune(Xtrain, ytrain, epochs=1)

    # predicting using the SDA
    pred = sDA.predict(Xtest).argmax(1)
    # let's see how the network did
    y = ytest.argmax(1)
    e0 = 0.0; y0 = len([0 for yi in range(len(y)) if y[yi]==0])
    e1 = 0.0; y1 = len([1 for yi in range(len(y)) if y[yi]==1])
    for i in range(len(y)):
        if(y[i] == 1):
            #print(y[i]==pred[i], y[i])
            e1 += y[i]==pred[i]
        if(y[i] == 0):
            print(y[i]==pred[i], y[i])
            e0 += y[i]==pred[i]

    # printing the result, this structure should result in 80% accuracy
    print "accuracy for class 0: %2.2f%%"%(100*e0/y0)
    print "accuracy for class 1: %2.2f%%"%(100*e1/y1)

    return sDA
    
    
if __name__ == '__main__':
    demo()
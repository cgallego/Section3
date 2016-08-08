# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:02:43 2016

@author: DeepLearning
"""

import itertools
import six.moves.cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline 

########
# Datasets sizes and class distributions
########
from DLFuncs_2D_temporal import *
funcs = DLFuncs_2D_temporal()
traindata_path= 'allLpatches_10x10x5.pklz' #'allLpatches_subs_smaller.pklz' #'allLpatches.pklz'
trainUdata_path= 'allUpatches_10x10x5.pklz'#'allUpatches_subs_smaller.pklz' #'allUpatches.pklz'
labeldata_path= 'allLabels_10x10x5.pklz' #'allLabels_subs_smaller.pklz' #'allLabels.pklz'

# read original data
datasets = funcs.load_wUdata(traindata_path, labeldata_path, trainUdata_path)

train_set_x, train_set_y = datasets[0]
np_train_x, np_train_y = datasets[3]
valid_set_x, valid_set_y = datasets[1]
np_valid_x, np_valid_y = datasets[4]        
test_set_x, test_set_y = datasets[2]
np_test_x, np_test_y = datasets[5]

# some stats, like datasets sizes and histograms
print "Training set size n= %d, vector size 10*10*5 = %d " % (np.asarray(np_train_x).shape)
print "Validation set size n= %d, vector size 10*10*5 = %d " % (np.asarray(np_valid_x).shape)
print "Test set size n= %d, vector size 10*10*5 = %d " % (np.asarray(np_test_x).shape)
print "==========\n Total datasets n = %d" % (np.asarray(np_train_x).shape[0]+np.asarray(np_valid_x).shape[0]+np.asarray(np_test_x).shape[0])
print "==========\n Training class 0 fractions = %f, class 1 = %f" % (float(np.sum(np.asarray(np_train_y)==0))/len(np_train_y), float(np.sum(np.asarray(np_train_y)==1))/len(np_train_y))
print "Validation class 0 fractions = %f, class 1 = %f" % (float(np.sum(np.asarray(np_valid_y)==0))/len(np_valid_y), float(np.sum(np.asarray(np_valid_y)==1))/len(np_valid_y))
print "Test set class 0 fractions = %f, class 1 = %f" % (float(np.sum(np.asarray(np_test_y)==0))/len(np_test_y), float(np.sum(np.asarray(np_test_y)==1))/len(np_test_y))

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
ax[0].hist(np.asarray(np_train_x).ravel(), bins=50, color='black')
ax[0].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax[0].set_xlabel('hist train datasets')
ax[1].hist(np.asarray(np_valid_x).ravel(), bins=50, color='black')
ax[1].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax[1].set_xlabel('hist valid datasets')
ax[2].hist(np.asarray(np_test_x).ravel(), bins=50, color='black')
ax[2].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax[2].set_xlabel('hist test datasets')
plt.show()

########
# Data trasnformation for pretaining SdA and finetunning for recognition of classes
########
# subtraction of pre contrast patch from post contrast patches
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

# replot of histograms
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
ax[0].hist(np.asarray(subsnp_train_x).ravel(), bins=50, color='black')
ax[0].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax[0].set_xlabel('hist subtracted train datasets')
ax[1].hist(np.asarray(subsnp_valid_x).ravel(), bins=50, color='black')
ax[1].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax[1].set_xlabel('hist subtracted valid datasets')
ax[2].hist(np.asarray(subsnp_test_x).ravel(), bins=50, color='black')
ax[2].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax[2].set_xlabel('hist subtracted test datasets')

########
# Grid search parameters iterated
########
pretrain_lr = [0.1]
noise_levels =[0.25,0.5] # [0.10, 0.25, 0.50]
nlayers = [1,2,3]
hidden_layers_sizes = [225,400,100,64,625]
hidden_layers_sidelen = [15,20,10,8,25] # [30,15,18,30]
batch_sizes = [50,10]

k=0
for item in itertools.product(nlayers, hidden_layers_sizes, noise_levels, pretrain_lr, batch_sizes): 
    k+=1
    print(k,item)
    
########
# Grid search parameters reasults
########
# best performance on validation (selected parameters)
pkl_filegridS = open('2D_temporal_gridSearch_results_10x10x5_subs.pkl','rb')
dfresults = pickle.load(pkl_filegridS)
print dfresults

avergAccu = (dfresults['Acuvalid0'])+np.asarray(dfresults['Acuvalid1'])
avergAccu = avergAccu/2
bestAccu = dfresults[avergAccu==np.max(avergAccu)]
print "best performing on Acuvalid parameters SdA"
print bestAccu, max(avergAccu)

# generalization accuracy on test set
avergAccu = (dfresults['Acutest0'])+np.asarray(dfresults['Acutest1'])
avergAccu = avergAccu/2
bestAccu = dfresults[avergAccu==np.max(avergAccu)]
print "best performing on Acutest parameters SdA"
print bestAccu, max(avergAccu)


########
# Use best performing model to classify some examples
########
with open('bestsDA_10x10x5_subs.obj', 'rb') as fp:
    bestsDA = pickle.load(fp)

# get first layer filters
Xtmp= bestsDA.dA_layers[0].W.get_value(borrow=True).T
imgX = Xtmp.reshape( Xtmp.shape[0], 4,10,10)[:,1,:,:]
image = tile_images(X=imgX, img_shape=(10, 10), 
                            tile_shape=(10, 10),
                            tile_spacing=(3, 3)) 
   
plt.figure()   
plt.imshow(image, cmap=plt.cm.gray)   

###############
# predicting using the SDA
###############
# take train case #1, class 1
Vol = subsnp_train_x[0].reshape(4,10,10)
imgslicestime = [Vol[0,:,:], Vol[1,:,:], Vol[2,:,:], Vol[3,:,:]]

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(12, 4))
ax[0,0].imshow(imgslicestime[3], cmap=plt.cm.gray)
ax[0,0].set_adjustable('box-forced')
ax[0,0].set_xlabel('a train case')
    
ax[0,1].hist(Vol.ravel(), bins=50, color='black')
ax[0,1].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax[0,1].set_xlabel('original')
    
tmp = subsnp_train_x[0]
# in this case just one layer
for L in bestsDA.sigmoid_layers:
    tmp = bestsDA.sigmoid_activate( tmp, L.W, L.b )
        
ax[0,2].imshow(tmp.reshape(15,15), cmap=plt.cm.gray)
ax[0,2].set_axis_off()
ax[0,2].set_adjustable('box-forced')
ax[0,3].hist(tmp.ravel(), bins=50, color='black')
ax[0,3].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax[0,3].set_xlabel('sDA activation histogram')
    
# finalize with log layer
predProb = bestsDA.softmax_activate( tmp, bestsDA.logLayer )
print "Label set class = %d ==> predProb = [%f,%f], pred class = %f" % (np_train_y[0],predProb[0],predProb[1],predProb.argmax() )

# take a train case #last one, class 0
Vol = subsnp_train_x[len(subsnp_train_x)-10].reshape(4,10,10)
imgslicestime = [Vol[0,:,:], Vol[1,:,:], Vol[2,:,:], Vol[3,:,:]]

ax[1,0].imshow(imgslicestime[3], cmap=plt.cm.gray)
ax[1,0].set_adjustable('box-forced')
ax[1,0].set_xlabel('a train case')
    
ax[1,1].hist(Vol.ravel(), bins=50, color='black')
ax[1,1].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax[1,1].set_xlabel('original')
    
tmp = subsnp_train_x[len(subsnp_train_x)-10]
# in this case just one layer
for L in bestsDA.sigmoid_layers:
    tmp = bestsDA.sigmoid_activate( tmp, L.W, L.b )
        
ax[1,2].imshow(tmp.reshape(15,15), cmap=plt.cm.gray)
ax[1,2].set_axis_off()
ax[1,2].set_adjustable('box-forced')
ax[1,3].hist(tmp.ravel(), bins=50, color='black')
ax[1,3].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax[1,3].set_xlabel('sDA activation histogram')
    
# finalize with log layer
predProb = bestsDA.softmax_activate( tmp, bestsDA.logLayer )
print "Label set class = %d ==> predProb = [%f,%f], pred class = %f" % (np_train_y[len(subsnp_train_x)-1],predProb[0],predProb[1],predProb.argmax() )



###############
# predicting using the SDA
###############
# take train case #1, class 1
Vol = subsnp_valid_x[0].reshape(4,10,10)
imgslicestime = [Vol[0,:,:], Vol[1,:,:], Vol[2,:,:], Vol[3,:,:]]

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(12, 4))
ax[0,0].imshow(imgslicestime[3], cmap=plt.cm.gray)
ax[0,0].set_adjustable('box-forced')
ax[0,0].set_xlabel('a train case')
    
ax[0,1].hist(Vol.ravel(), bins=50, color='black')
ax[0,1].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax[0,1].set_xlabel('original')
    
tmp = subsnp_valid_x[0]
# in this case just one layer
for L in bestsDA.sigmoid_layers:
    tmp = bestsDA.sigmoid_activate( tmp, L.W, L.b )
        
ax[0,2].imshow(tmp.reshape(15,15), cmap=plt.cm.gray)
ax[0,2].set_axis_off()
ax[0,2].set_adjustable('box-forced')
ax[0,3].hist(tmp.ravel(), bins=50, color='black')
ax[0,3].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax[0,3].set_xlabel('sDA activation histogram')
    
# finalize with log layer
predProb = bestsDA.softmax_activate( tmp, bestsDA.logLayer )
print "Label set class = %d ==> predProb = [%f,%f], pred class = %f" % (np_valid_y[0],predProb[0],predProb[1],predProb.argmax() )

# take a train case #last one, class 0
Vol = subsnp_valid_x[len(subsnp_valid_x)-10].reshape(4,10,10)
imgslicestime = [Vol[0,:,:], Vol[1,:,:], Vol[2,:,:], Vol[3,:,:]]

ax[1,0].imshow(imgslicestime[3], cmap=plt.cm.gray)
ax[1,0].set_adjustable('box-forced')
ax[1,0].set_xlabel('a train case')
    
ax[1,1].hist(Vol.ravel(), bins=50, color='black')
ax[1,1].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax[1,1].set_xlabel('original')
    
tmp = subsnp_valid_x[len(subsnp_valid_x)-10]
# in this case just one layer
for L in bestsDA.sigmoid_layers:
    tmp = bestsDA.sigmoid_activate( tmp, L.W, L.b )
        
ax[1,2].imshow(tmp.reshape(15,15), cmap=plt.cm.gray)
ax[1,2].set_axis_off()
ax[1,2].set_adjustable('box-forced')
ax[1,3].hist(tmp.ravel(), bins=50, color='black')
ax[1,3].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax[1,3].set_xlabel('sDA activation histogram')
    
# finalize with log layer
predProb = bestsDA.softmax_activate( tmp, bestsDA.logLayer )
print "Label set class = %d ==> predProb = [%f,%f], pred class = %f" % (np_valid_y[len(subsnp_valid_x)-1],predProb[0],predProb[1],predProb.argmax() )


###############
# predicting using the SDA
###############
# take train case #1, class 1
Vol = subsnp_test_x[0].reshape(4,10,10)
imgslicestime = [Vol[0,:,:], Vol[1,:,:], Vol[2,:,:], Vol[3,:,:]]

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(12, 4))
ax[0,0].imshow(imgslicestime[3], cmap=plt.cm.gray)
ax[0,0].set_adjustable('box-forced')
ax[0,0].set_xlabel('a train case')
    
ax[0,1].hist(Vol.ravel(), bins=50, color='black')
ax[0,1].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax[0,1].set_xlabel('original')
    
tmp = subsnp_test_x[0]
# in this case just one layer
for L in bestsDA.sigmoid_layers:
    tmp = bestsDA.sigmoid_activate( tmp, L.W, L.b )
        
ax[0,2].imshow(tmp.reshape(15,15), cmap=plt.cm.gray)
ax[0,2].set_axis_off()
ax[0,2].set_adjustable('box-forced')
ax[0,3].hist(tmp.ravel(), bins=50, color='black')
ax[0,3].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax[0,3].set_xlabel('sDA activation histogram')
    
# finalize with log layer
predProb = bestsDA.softmax_activate( tmp, bestsDA.logLayer )
print "Label set class = %d ==> predProb = [%f,%f], pred class = %f" % (np_test_y[0],predProb[0],predProb[1],predProb.argmax() )

# take a train case #last one, class 0
Vol = subsnp_test_x[len(subsnp_test_x)-10].reshape(4,10,10)
imgslicestime = [Vol[0,:,:], Vol[1,:,:], Vol[2,:,:], Vol[3,:,:]]

ax[1,0].imshow(imgslicestime[3], cmap=plt.cm.gray)
ax[1,0].set_adjustable('box-forced')
ax[1,0].set_xlabel('a train case')
    
ax[1,1].hist(Vol.ravel(), bins=50, color='black')
ax[1,1].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax[1,1].set_xlabel('original')
    
tmp = subsnp_test_x[len(subsnp_test_x)-10]
# in this case just one layer
for L in bestsDA.sigmoid_layers:
    tmp = bestsDA.sigmoid_activate( tmp, L.W, L.b )
        
ax[1,2].imshow(tmp.reshape(15,15), cmap=plt.cm.gray)
ax[1,2].set_axis_off()
ax[1,2].set_adjustable('box-forced')
ax[1,3].hist(tmp.ravel(), bins=50, color='black')
ax[1,3].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax[1,3].set_xlabel('sDA activation histogram')
    
# finalize with log layer
predProb = bestsDA.softmax_activate( tmp, bestsDA.logLayer )
print "Label set class = %d ==> predProb = [%f,%f], pred class = %f" % (np_test_y[len(subsnp_test_x)-1],predProb[0],predProb[1],predProb.argmax() )

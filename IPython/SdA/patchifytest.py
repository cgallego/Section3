# -*- coding: utf-8 -*-
"""
Created on Mon May 09 12:22:14 2016

@author: Cristina Gallego
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
import math

from skimage import data, img_as_float
from skimage import exposure
import matplotlib.patches as mpatches
import matplotlib.ticker as plticker

from StackeddAutoencoder import *

import six.moves.cPickle as pickle
from utils import tile_raster_images
try:
    import PIL.Image as Image
except ImportError:
    import Image

#!/usr/bin/env python
class Patchifytest(object):
    """Imgage Patchify coordinates and extract Patches functions on test imgae"""
    
    def __init__(self):
        self.ptcsize = None
        self.img_size = None

    def __call__(self):       
        """ Turn Class into a callable object """
        Patchifytest()
        
    def contrast_stretch(self, img):  
        ## Floating point images are between 0 and 1 (unsigned images) or -1 and 1 (signed images), 
        # while 8-bit images are expected to have values in {0,255}.
        #imgorig = img.view(np.uint8)
        p2, p999 = np.percentile(img, (2, 99.9))
        # Contrast stretching
        img_rescale = exposure.rescale_intensity(img, in_range=(p2, p999))
        # rescale to values again between o and 1 (unsigned images) 
        #img_rescale = imgres.view(np.float32)
        
        return img_rescale
        
            
    def extractPatches(self, imgArray, x1, y1, x2, y2, name, snapshot_loc):
        """
        Currently only supports rectangular patches
        extractPatch takes pixel coordinates of 2 points defining patch diagonals
        e.g: x1, y1, x2, y2
        
        Usage:
        # plot slice location over time
        extrPatch = Patchify()
        extrPatch.extractPatch(imgArray, x1, y1, x2, y2)
        
        imgArray:: contains preimg, img1 to img4 (total 5 slices)
        """
        # get img size
        self.img_size = imgArray[0].shape
        self.ptcsize = imgArray[0][x1:x2,y1:y2].shape # patch1.flatten().reshape(64L, 52L) == patch1
        
        patches = []
        # show       
        fig, axes = plt.subplots(ncols=2, nrows=5, figsize=(4, 4))
        a = axes.flat 
        
        # finally append all elemnet arrays
        allpatches = []
        
        ## for the post-contract imgs
        for k in range(0,len(imgArray)):
            # extract patch inside the rectangular ROI
            patchk = imgArray[k][x1:x2,y1:y2]
            patches.append( patchk.flatten() ) 
            a[2*k].imshow(imgArray[k], cmap=plt.cm.gray) 
            
            allpatches = np.insert(allpatches, len(allpatches), patches[k])
            # reshape and extract
            eimg = allpatches.reshape(k+1, self.ptcsize[1], self.ptcsize[0])
            img = eimg[k,:,:]
            eslice = img.reshape(self.ptcsize[0],self.ptcsize[1])
            #plot patch
            a[2*k+1].imshow(eslice, cmap=plt.cm.gray) 
            
        # display
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.savefig(snapshot_loc+os.sep+name+'.pdf')
        plt.show(block=True) 
        
        return allpatches
        
    def tile_images(self, X, img_shape, tile_shape, tile_spacing=(0, 0)):
        """
        Transform an array with one flattened image per row, into an array in
        which images are reshaped and layed out like tiles on a floor.
    
        This function is useful for visualizing datasets whose rows are images,
        and also columns of matrices for transforming those rows
        (such as the first layer of a neural net).
    
        :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
        be 2-D ndarrays or None;
        :param X: a 2-D array in which every row is a flattened image.
    
        :type img_shape: tuple; (height, width)
        :param img_shape: the original shape of each image
    
        :type tile_shape: tuple; (rows, cols)
        :param tile_shape: the number of images to tile (rows, cols)    
    
        :returns: array suitable for viewing as an image.
        (See:`Image.fromarray`.)
        :rtype: a 2-d array with same dtype as X.
    
        """
    
        assert len(img_shape) == 2
        assert len(tile_shape) == 2
        assert len(tile_spacing) == 2
    
        out_shape = [
            (ishp + tsp) * tshp - tsp
            for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
        ]
    
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
                    
        return out_array
     
   
    def extractROIPatches(self, pimg1, pimg2, pimg3, pimg4, centroid, patch_size, 
                          patch_diag1, patch_diag2, ha, wa, name):
        """
        Currently only supports rectangular patches
        extractPatch takes pixel coordinates of 2 points defining patch diagonals
        e.g: x1, y1, x2, y2
        
        Usage:
        # plot slice location over time
        extrPatch = Patchify()
        extrPatch.extractPatch(imgArray, x1, y1, x2, y2)
        
        imgArray:: contains preimg, img1 to img4 (total 5 slices)
        """
                
        # get patches sizes and quatities
        sno = int(centroid[2])-1
        x0 = float(centroid[0])
        y0 = float(centroid[1])
        hp = float(long(patch_size[0]))
        wp = float(long(patch_size[1]))
        self.optcsize = [hp-1, wp-1]
        self.ptcsize = [ha, wa]
        
        x1 = float(patch_diag1[0])
        y1 = float(patch_diag1[1])
        x2 = float(patch_diag2[0])
        y2 = float(patch_diag2[1])
        if(x1<x2):
            x1 = x1-10
            x2 = x2+10
        if(x1>x2):
            xt = x1
            x1 = x2-10
            x2 = xt+10
        if(y1<y2):
            y1 = y1-10
            y2 = y2+10
        if(y1>y2):
            yt = y1
            y1 = y2-10
            y2 = yt+10
        
        print '[x1:x2][y1:y2] [%d,%d][%d,%d]' % (x1,x2,y1,y2)
        x0 = x1 + (x2-x1)/2
        y0 = y1 + (y2-y1)/2
        z0 = sno
        print 'indexed centroid:', x0,y0,z0
        
        # derive radius of proportional samplling
        ra = np.sqrt(ha**2 + wa**2)/2
        rp = np.sqrt((hp-1)**2 + (wp-1)**2)/2
        rationp = min(int(round(rp/ra)),3) # to limit repetition to not fall outside imag
        skewp = wp/hp
        
        print("====================")
        print 'ra, rp, rationp, skewp:', ra,rp,rationp, skewp
        print("====================")
        
        # will sample np times in each directions
        # from centroid x0,y0 to a radius distance of  ra (hypothenuse of triangle)
        directions = [0, math.radians(45), math.radians(90), math.radians(135), math.radians(180), math.radians(225), math.radians(270), math.radians(315)]
        #[0, math.radians(45), math.radians(90), math.radians(135), math.radians(180), math.radians(225), math.radians(270), math.radians(315)]
        npatches = len(directions)*rationp+1
        print 'sampling %d times, for a total of %d patches.' % (rationp, npatches)
        xs=[]
        ys=[]
        allLpatches = []

        # show       
        fig, ax = plt.subplots(nrows=4, ncols=rationp*len(directions)+2, figsize=(4, 4))
        
        ## for the post-contract imgs
        # samples from image at xs, ys locations
        imgslicestime = [ pimg1[sno,:,:], pimg2[sno,:,:], pimg3[sno,:,:], pimg4[sno,:,:]]
        
        # init with a patch size centered in the orig patch
        xinit = (x0-ra/2)  
        yinit = (y0-ra/2)  
        centeredpatches = []
        minO = 5
        
        # extract patch inside the rectangular ROI
        for kimg in range(0,len(imgslicestime)):   
            patchk = imgslicestime[kimg][xinit:xinit+self.ptcsize[0],yinit:yinit+self.ptcsize[1]]  
            # add centered patch only once, the first patches
            centeredpatches = np.insert(centeredpatches, len(centeredpatches), patchk.flatten())
            
        # append extracted patch of size npratio * 8directions (30x30x4) and append 
        allLpatches.append(centeredpatches)        
      
        kp=2  
        for k in range(1,rationp+1):
            # per each k repeat a sample per each direction
            print 'sampling %d times......' % (k)
            for angle,j in zip(directions,range(len(directions))): 
                
                x = (x0-ra/2) - k*wa*rationp/minO*np.cos(angle)*wp/hp 
                y = (y0-ra/2) - k*ha*rationp/minO*np.sin(angle)*hp/wp
                
                # append and continue
                print 'sampling %f angle, (cos=%f, sin=%f), patch Origin: %s' % (int(math.degrees(angle)), np.cos(angle), np.sin(angle), str([x,y]) )
                xs.append(x)
                ys.append(y)
                
                # append all elemnet arrays
                allsubpatches = []
                
                for kimg in range(0,len(imgslicestime)):                    
                    print 'sampling image post-contrast time-point %d...... [-x %f,-y %f]' % ((kimg+1), k*wa*rationp/minO*np.cos(angle)*wp/hp , k*ha*rationp/minO*np.sin(angle)*hp/wp )
                    # extract patch inside the rectangular ROI
                    patchk = imgslicestime[kimg][x1:x2,y1:y2]                
                    #plot original patch
                    ax[kimg,0].imshow(patchk, cmap=plt.cm.gray) 
                    ax[kimg,0].set_title('imgslicestime,\ntime='+str(kimg+1), fontsize=10)
                    xmajor_ticks = [int(xi) for xi in [x1,x2]]
                    ymajor_ticks = [int(yi) for yi in [y1,y2]]
                    ax[kimg,0].set_xlabel(str(xmajor_ticks)) 
                    ax[kimg,0].set_ylabel(str(ymajor_ticks))
                                                    
                    #plot original patch
                    ax[kimg,1].imshow(patchk, cmap=plt.cm.gray) 
                    ax[kimg,1].set_title('centered patch', fontsize=10)
                    xmajor_ticks = [int(xi) for xi in [xinit,xinit+self.ptcsize[0]]]
                    ymajor_ticks = [int(yi) for yi in [yinit,yinit+self.ptcsize[1]]]
                    ax[kimg,1].set_xlabel(str(xmajor_ticks)) 
                    ax[kimg,1].set_ylabel(str(ymajor_ticks))
                                                   
                    # finally append all elemnet arrays
                    xp1 = int(x)
                    yp1 = int(y)
                    xp2 = int(x + self.ptcsize[0])
                    yp2 = int(y + self.ptcsize[1])
                
                    # subpatch
                    subpatchk = imgslicestime[kimg][xp1:xp2,yp1:yp2]
                    allsubpatches = np.insert(allsubpatches, len(allsubpatches), subpatchk.flatten())
                    
                    #plot subpatch
                    ax[kimg,kp].imshow(subpatchk, cmap=plt.cm.gray) 
                    ax[kimg,kp].set_title(str(k)+'_da_'+str(int(math.degrees(angle))), fontsize=10)
                    
                    xpmajor_ticks = [int(xi) for xi in [xp1,xp2]]
                    ypmajor_ticks = [int(yi) for yi in [yp1,yp2]]
                    
                    ax[kimg,kp].set_xlabel(str(xpmajor_ticks)) 
                    ax[kimg,kp].set_ylabel(str(ypmajor_ticks))
            
                # append extracted patch of size npratio * 8directions (30x30x4) and append 
                allLpatches.append(allsubpatches)
                kp += 1
                print len(allLpatches)
                
        # display
        # Fine-tune figure; make subplots close to each other and hide x ticks for
        # all but bottom plot.
        plt.setp([a.get_xticklabels() for a in fig.axes], visible=False)
        plt.setp([a.get_yticklabels() for a in fig.axes], visible=False)
        
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        
        fig.tight_layout()
        plt.savefig(snapshot_loc+os.sep+name+'.pdf')
        plt.show(block=False) 
        
        plt.close()
                
        return allLpatches

       
    def extractestPatches(self, slicestime, ha, wa, stride, name):
        """
        Currently only supports rectangular patches
        extractPatch takes pimg0, pimg1, pimg2, pimg3, pimg4 vol arrays and a list of selected slices, 
        selected x,y init patch locations
        
        Usage:
        # plots patches location over time
        extrPatch = Patchify()
        extrPatch.extractUnlabeledPatches(pimg0, pimg1, pimg2, pimg3, pimg4, selectedsl, selectednx, selectedny, name):
        
        preimg, img1 to img4 (total 5 slices)
        self.ptcsize = imgslicestime[0][x1:x2,y1:y2].shape # patch1.flatten().reshape(64L, 52L) == patch1
        
        """        
        self.ptcsize = [ha,wa]
        with open('bestsDA_10x10x5_subs.obj', 'rb') as fp:
            bestsDA = pickle.load(fp)
        
        volsize = slicestime[0].shape
        ncols = volsize[1]
        nrows = volsize[2]
        nslices = volsize[0]
        
        # to standarize the data. The goal of this process is to scale the inputs 
        # to have mean 0 and a variance of 1. In this case, you need to substract the mean 
        # value of the column and divide by the standard deviation:
        #        let's think a bit about the rationale behind an auto-encoder (AE):
        #        The purpose of auto-encoder is to learn, in an unsupervised manner, something about the underlying structure of the input data. How does AE achieves this goal? If it manages to reconstruct the input signal from its output signal (that is usually of lower dimension) it means that it did not lost information and it effectively managed to learn a more compact representation.
        #        
        #        In most examples, it is assumed, for simplicity, that both input signal and output signal ranges in [0..1]. Therefore, the same non-linearity (sigmf) is applied both for obtaining the output signal and for reconstructing back the inputs from the outputs.
        #        Something like
        #        
        #        output = sigmf( W*input + b ); % compute output signal
        #        reconstruct = sigmf( W'*output + b_prime ); % notice the different constant b_prime
        #        Then the AE learning stage tries to minimize the training error || output - reconstruct ||.
        #        
        #        However, who said the reconstruction non-linearity must be identical to the one used for computing the output?
        #        In your case, the assumption that inputs ranges in [0..1] does not hold. Therefore, it seems that you need to use a different non-linearity for the reconstruction. You should pick one that agrees with the actual range of you inputs.
        #        If, for example, your input ranges in (0..inf) you may consider using exp or ().^2 as the reconstruction non-linearity. You may use polynomials of various degrees, log or whatever function you think may fit the spread of your input data.
        #        
        normimgslicestime = []
        subsnormimgslicestime = []
        for k in range(len(slicestime)):
            Vol = np.asarray(slicestime[k])
            muVol = np.mean(Vol)
            varVol = np.var(Vol)
            print(muVol,varVol)
            normVol = (Vol - muVol)/np.sqrt(varVol+0.00001)
            normimgslicestime.append(normVol)
            ## implement patch substraction
            if(k>0):
                subsnormimgslicestime.append(normVol - normimgslicestime[0])
        
        alltestpatches = []  
        
        for k in range(nslices):
            allslicepatches = []
            npatches_holder = []
            
            ###########
            # extract slice
            # implement slice based substraction     
            ###########
            #subVol1 = self.contrast_stretch( slicestime[1][k,:,:]-slicestime[0][k,:,:] )
            #subVol2 = self.contrast_stretch( slicestime[2][k,:,:]-slicestime[0][k,:,:] )
            #subVol3 = self.contrast_stretch( slicestime[3][k,:,:]-slicestime[0][k,:,:] )
            #subVol4 = self.contrast_stretch( slicestime[4][k,:,:]-slicestime[0][k,:,:] ) 
            #plt.imshow(pimg4, cmap=plt.cm.gray) 
                
            # allslicepatches to collect
            #proslicestime = [subVol1, subVol2, subVol3, subVol4]
            
            print" \n==================\n Classifying slice no %i...\n==================" % k
            for i in range(ncols/stride):
                for j in range(nrows/stride): # npatches =17*17=289
                    istissue = True
                    rightS = False
                    # select nx, ny for patches
                    x1 = i*stride
                    x2 = i*stride+ha
                    y1 = j*stride
                    y2 = j*stride+wa
                    
                    stridepatches = []
                    ## for the post-contract imgs
                    for l in range(len(subsnormimgslicestime)): 
                        
                        # extract patch inside the rectangular ROI
                        patchk = subsnormimgslicestime[l][k,x1:x2,y1:y2]
                    
                        # check patch is right size
                        if(len(patchk.flatten()) == ha*wa):
                            rightS = True
                            stridepatches = np.insert(stridepatches, len(stridepatches), patchk.flatten())

                    # append extracted patch of size 3600L (30x30x4) and append a total of nrowsxcols = 50 
                    if(rightS and istissue):
                        allslicepatches.append(stridepatches)
                        npatches_holder.append(True)
            
            alltestpatches.append(allslicepatches) 
            
            ###############                                
            # format
            ###############
            Xtmp = np.asarray(allslicepatches)
            imgX = Xtmp.reshape( Xtmp.shape[0], 4, 100)[:,3,:]
            pathcesimgX = self.tile_images(X=imgX , img_shape=(10, 10), 
                               tile_shape=(51, 51),
                               tile_spacing=(1, 1))
                 
            positive_patches = []
            label_cascadepatches = []
            flag_output = np.zeros([ncols/stride,nrows/stride], dtype='UInt8')   
            c=0; cp=0;
            for i in range(ncols/stride):
                for j in range(nrows/stride):
                    if(npatches_holder[c]):
                        patch = allslicepatches[cp]
                        ###############
                        # predicting using the SDA 
                        # np.array([1,0]).argmax(0) is a 0 prediction
                        # np.array([0,1]).argmax(0) is a 1 prediction
                        ###############
                        # in train
                        predtrain = bestsDA.predict_functions(patch)
                        #print "pred = [%f,%f], argmax is class %d " % (predtrain[0],predtrain[1],predtrain.argmax(0))
                        flag_output[i,j] = predtrain.argmax(0)
                        if(predtrain.argmax(0) == 1):
                            positive_patches.append(patch)
                            label_cascadepatches.append(0)
                            

                        # augment one for patch
                        cp+=1
                    # augment one for flag
                    c+=1
            
            clasfslicepatches = []
            sumofpositives = 0
            for i in range(ncols/stride):
                for j in range(nrows/stride): # npatches =17*17=289
                    # select nx, ny for patches
                    x1 = i*stride
                    x2 = i*stride+ha
                    y1 = j*stride
                    y2 = j*stride+wa
                    
                    stridepatches = []
                    ## for the post-contract imgs
                    for l in range(len(subsnormimgslicestime)): 
                        
                        # extract patch inside the rectangular ROI
                        patchk = subsnormimgslicestime[l][k,x1:x2,y1:y2]
                    
                        # check patch is right size
                        if(len(patchk.flatten()) == ha*wa):
                            rightS = True
                            stridepatches = np.insert(stridepatches, len(stridepatches), patchk.flatten())

                    # append extracted patch of size 3600L (30x30x4) and append a total of nrowsxcols = 50 
                    if(flag_output[i,j]):
                        clasfslicepatches.append(stridepatches)
                        sumofpositives += 1
                    else:
                        clasfslicepatches.append( np.zeros(4*10*10) )
                        
            ###############                                
            # display per slide
            ###############
            Xtmp = np.asarray(clasfslicepatches)
            imgX = Xtmp.reshape( Xtmp.shape[0], 4, 100)[:,3,:]
            clasfpathcesimgX = self.tile_images(X=imgX , img_shape=(10, 10), 
                               tile_shape=(51, 51),
                               tile_spacing=(1, 1))
                               
            # show       
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 12))
            ax[0].imshow( pathcesimgX, cmap=plt.cm.gray ) 
            ax[1].imshow( flag_output, cmap=plt.cm.gray )
            ax[2].imshow( clasfpathcesimgX, cmap=plt.cm.gray )
            plt.show()
            
            print "done plotting positives = %i" % sumofpositives

            #mng = plt.get_current_fig_manager()
            #mng.window.showMaximized()
            #fig.tight_layout()
            #plt.show(block=True)            
            #plt.close()
        
        return alltestpatches        
        

    def  extractestPatches_sliceSelect(self, slicestime, ha, wa, stride, name, centroid):
        """
        Currently only supports rectangular patches
        extractPatch takes pimg0, pimg1, pimg2, pimg3, pimg4 vol arrays and a list of selected slices, 
        selected x,y init patch locations
        
        Usage:
        # plots patches location over time
        extrPatch = Patchify()
        extrPatch.extractUnlabeledPatches(pimg0, pimg1, pimg2, pimg3, pimg4, selectedsl, selectednx, selectedny, name):
        
        preimg, img1 to img4 (total 5 slices)
        self.ptcsize = imgslicestime[0][x1:x2,y1:y2].shape # patch1.flatten().reshape(64L, 52L) == patch1
        
        """   
        [centroid_slice, centroid_y, centroid_x] = centroid
        
        self.ptcsize = [ha,wa]
        with open('bestsDA_10x10x5_subs.obj', 'rb') as fp:
            bestsDA = pickle.load(fp)
        
        volsize = slicestime[0].shape
        ncols = volsize[1]
        nrows = volsize[2]
        nslices = volsize[0]
        
        # to standarize the data. The goal of this process is to scale the inputs 
        # to have mean 0 and a variance of 1. In this case, you need to substract the mean 
        # value of the column and divide by the standard deviation:
        #        let's think a bit about the rationale behind an auto-encoder (AE):
        #        The purpose of auto-encoder is to learn, in an unsupervised manner, something about the underlying structure of the input data. How does AE achieves this goal? If it manages to reconstruct the input signal from its output signal (that is usually of lower dimension) it means that it did not lost information and it effectively managed to learn a more compact representation.
        #        
        #        In most examples, it is assumed, for simplicity, that both input signal and output signal ranges in [0..1]. Therefore, the same non-linearity (sigmf) is applied both for obtaining the output signal and for reconstructing back the inputs from the outputs.
        #        Something like
        #        
        #        output = sigmf( W*input + b ); % compute output signal
        #        reconstruct = sigmf( W'*output + b_prime ); % notice the different constant b_prime
        #        Then the AE learning stage tries to minimize the training error || output - reconstruct ||.
        #        
        #        However, who said the reconstruction non-linearity must be identical to the one used for computing the output?
        #        In your case, the assumption that inputs ranges in [0..1] does not hold. Therefore, it seems that you need to use a different non-linearity for the reconstruction. You should pick one that agrees with the actual range of you inputs.
        #        If, for example, your input ranges in (0..inf) you may consider using exp or ().^2 as the reconstruction non-linearity. You may use polynomials of various degrees, log or whatever function you think may fit the spread of your input data.
        #
        normimgslicestime = []
        subsnormimgslicestime = []
        for k in range(len(slicestime)):
            Vol = np.asarray(slicestime[k])
            muVol = np.mean(Vol)
            varVol = np.var(Vol)
            print(muVol,varVol)
            normVol = (Vol - muVol)/np.sqrt(varVol+0.00001)
            normimgslicestime.append(normVol)
            ## implement patch substraction
            if(k>0):
                subsnormimgslicestime.append(normVol - normimgslicestime[0])
        

        allslicepatches = []
        npatches_holder = []
        print" \n==================\n Classifying slice no %i...\n==================" % centroid_slice
        for i in range(ncols/stride):
            for j in range(nrows/stride): # npatches =17*17=289
                istissue = True
                rightS = False
                # select nx, ny for patches
                x1 = i*stride
                x2 = i*stride+ha
                y1 = j*stride
                y2 = j*stride+wa
                
                stridepatches = []
                ## for the post-contract imgs
                for l in range(len(subsnormimgslicestime)): 
                    
                    # extract patch inside the rectangular ROI
                    patchk = subsnormimgslicestime[l][centroid_slice,x1:x2,y1:y2]
                
                    # check patch is right size
                    if(len(patchk.flatten()) == ha*wa):
                        rightS = True
                        stridepatches = np.insert(stridepatches, len(stridepatches), patchk.flatten())

                # append extracted patch of size 3600L (30x30x4) and append a total of nrowsxcols = 50 
                if(rightS and istissue):
                    allslicepatches.append(stridepatches)
                    npatches_holder.append(True)
                        
            ###############                                
            # format
            ###############
            Xtmp = np.asarray(allslicepatches)
            imgX = Xtmp.reshape( Xtmp.shape[0], 4, 100)[:,3,:]
            pathcesimgX = self.tile_images(X=imgX , img_shape=(10, 10), 
                               tile_shape=(51, 51),
                               tile_spacing=(1, 1))
                               
            flag_output = np.zeros([ncols/stride,nrows/stride], dtype='UInt8')   
            c=0; cp=0;
            for i in range(ncols/stride):
                for j in range(nrows/stride):
                    if(npatches_holder[c]):
                        patch = allslicepatches[cp]
                        ###############
                        # predicting using the SDA 
                        # np.array([1,0]).argmax(0) is a 0 prediction
                        # np.array([0,1]).argmax(0) is a 1 prediction
                        ###############
                        # in train
                        predtrain = bestsDA.predict_functions(patch)
                        #print "pred = [%f,%f], argmax is class %d " % (predtrain[0],predtrain[1],predtrain.argmax(0))
                        flag_output[i,j] = predtrain.argmax(0)
                        # augment one for patch
                        cp+=1
                    # augment one for flag
                    c+=1
            
            clasfslicepatches = []
            sumofpositives = 0
            for i in range(ncols/stride):
                for j in range(nrows/stride): # npatches =17*17=289
                    # select nx, ny for patches
                    x1 = i*stride
                    x2 = i*stride+ha
                    y1 = j*stride
                    y2 = j*stride+wa
                    
                    stridepatches = []
                    ## for the post-contract imgs
                    for l in range(len(subsnormimgslicestime)): 
                        
                        # extract patch inside the rectangular ROI
                        patchk = subsnormimgslicestime[l][centroid_slice,x1:x2,y1:y2]
                    
                        # check patch is right size
                        if(len(patchk.flatten()) == ha*wa):
                            rightS = True
                            stridepatches = np.insert(stridepatches, len(stridepatches), patchk.flatten())

                    # append extracted patch of size 3600L (30x30x4) and append a total of nrowsxcols = 50 
                    if(flag_output[i,j]):
                        clasfslicepatches.append(stridepatches)
                        sumofpositives += 1
                    else:
                        clasfslicepatches.append( np.zeros(4*10*10) )
                        
        ###############                                
        # display per slide
        ###############
        Xtmp = np.asarray(clasfslicepatches)
        imgX = Xtmp.reshape( Xtmp.shape[0], 4, 100)[:,3,:]
        clasfpathcesimgX = self.tile_images(X=imgX , img_shape=(10, 10), 
                           tile_shape=(51, 51),
                           tile_spacing=(1, 1))
                           
        # show       
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 12))
        ax[0].imshow( pathcesimgX, cmap=plt.cm.gray ) 
        ax[1].imshow( flag_output, cmap=plt.cm.gray )
        ax[2].imshow( clasfpathcesimgX, cmap=plt.cm.gray )
        ax[2].annotate("", xy=(centroid_y, centroid_x), xycoords='data',
                                xytext=(centroid_y+10, centroid_x+10), textcoords='data',
                                arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
        plt.show()
        
        print "done plotting positives = %i" % sumofpositives

        
        return alltestpatches    
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:50:00 2016

@author: DeepLearning
"""

import os, os.path
import fnmatch
import SimpleITK as sitk

import six.moves.cPickle as pickle
import gzip
import seaborn as sns
import pandas as pd
import numpy as np
import itertools

from patchifytest import *


def filen_patt_match(mha_lesion_loc, patternstr):
    
    selfilenm = []
    for filenm in os.listdir(mha_lesion_loc):
        if fnmatch.fnmatch(filenm, patternstr):
            selfilenm.append(filenm)
        
    return selfilenm
    
def run_mha_lesion(mha_lesion_loc, PatientID, AccessionN):
    
    ## Format query StudyID
    print "Reading volumnes..."
     
    ## read Image (right now needs to use filename pattern matching)
    fpttrn = PatientID+'_'+AccessionN+'_*'
    files_pttrn = filen_patt_match(mha_lesion_loc, fpttrn)
    files_Vols=[]
    for k in range(len(files_pttrn)):
        if('@' in files_pttrn[k]):
            files_Vols.append(files_pttrn[k])
    
    # find the lowest numbered series as the pre-contrast
    DynSeriesids = []
    for item in files_Vols:
        DynSeriesids.append( int(item[item.find('_')+1:item.find('@')][-3::]) )  
            
    Dynid = min(DynSeriesids)
    
    vol0f = PatientID+'_'+AccessionN+'_'+str(int(Dynid))+'@*'
    filen_Vol0 = filen_patt_match(mha_lesion_loc, vol0f)
    Vol0 = sitk.ReadImage(mha_lesion_loc+os.sep+filen_Vol0[0])
    
    vol1f = PatientID+'_'+AccessionN+'_'+str(int(Dynid)+1)+'@*'
    filen_Vol1 = filen_patt_match(mha_lesion_loc, vol1f)
    Vol1 = sitk.ReadImage(mha_lesion_loc+os.sep+filen_Vol1[0])
    
    vol2f = PatientID+'_'+AccessionN+'_'+str(int(Dynid)+2)+'@*'
    filen_Vol2 = filen_patt_match(mha_lesion_loc, vol2f)
    Vol2 = sitk.ReadImage(mha_lesion_loc+os.sep+filen_Vol2[0])
    
    vol3f = PatientID+'_'+AccessionN+'_'+str(int(Dynid)+3)+'@*'
    filen_Vol3 = filen_patt_match(mha_lesion_loc, vol3f)
    Vol3 = sitk.ReadImage(mha_lesion_loc+os.sep+filen_Vol3[0])
    
    vol4f = PatientID+'_'+AccessionN+'_'+str(int(Dynid)+4)+'@*'
    filen_Vol4 = filen_patt_match(mha_lesion_loc, vol4f)
    Vol4 = sitk.ReadImage(mha_lesion_loc+os.sep+filen_Vol4[0])

    # reformat Vol slices as Float32
    Vol0 = sitk.GetArrayFromImage(sitk.Cast(Vol0,sitk.sitkFloat32))
    Vol1 = sitk.GetArrayFromImage(sitk.Cast(Vol1,sitk.sitkFloat32))
    Vol2 = sitk.GetArrayFromImage(sitk.Cast(Vol2,sitk.sitkFloat32))
    Vol3 = sitk.GetArrayFromImage(sitk.Cast(Vol3,sitk.sitkFloat32))
    Vol4 = sitk.GetArrayFromImage(sitk.Cast(Vol4,sitk.sitkFloat32))

    
    # list post contrast volumes usually Vol1.GetSize() = (512, 512, 88)
    slicestime = [ Vol0,
                   Vol1, 
                   Vol2,
                   Vol3, 
                   Vol4 ]  
                   
    return slicestime
    

   
if __name__ == '__main__':
    # Get Root folder ( the directory of the script being run)
    path_rootFolder = os.path.dirname(os.path.abspath(__file__))
    
    # start by reading mha imgaes
    mha_lesion_loc = 'Z:/Cristina/Section3/mha'
    topatched_mris = 'view_allpatched_mris_wdiagonals.csv'
    output_folder = 'plots_firstSdApatches'

    # Open filename patched_mris
    patched_ids = pd.read_csv(topatched_mris)
#    all_positive_patches = []
#    all_label_cascadepatches = []
    
    # read 
    with gzip.open(output_folder+'/patches_firstSdA.pklz', 'rb') as f:
        try:
            all_positive_patches = pickle.load(f, encoding='latin1')
        except:
            all_positive_patches = pickle.load(f)

    with gzip.open(output_folder+'/all_labels_firstSdA.pklz', 'rb') as f:
        try:
            all_label_cascadepatches = pickle.load(f, encoding='latin1')
        except:
            all_label_cascadepatches = pickle.load(f)
    
    
    for kline in range(38,len(patched_ids)): 
        
        print(patched_ids.iloc[kline])
        
        # Get the line: Study#, DicomExam#
        lesion_id = patched_ids.iloc[kline]['lesion_id']
        PatientID = patched_ids.iloc[kline]['cad_pt_no_txt'] 

        # continue with patient not already patched
        AccessionN = patched_ids.iloc[kline]['exam_a_number_txt'] 
        dateID = patched_ids.iloc[kline]['exam_dt_datetime']  #dateID = '2010-11-29' as '2010,11,29';
        side = patched_ids.iloc[kline]['side_int']
        centroid = patched_ids.iloc[kline]['centroid']
        str_coords = pd.Series(centroid).str[1:-1].str.split(', ')
        centroid = str_coords.apply(pd.Series).astype(float)
        
        patch_diag1 = patched_ids.iloc[kline]['patch_diag1']
        str_coords = pd.Series(patch_diag1).str[1:-1].str.split(', ')
        diag1 = str_coords.apply(pd.Series).astype(float)
        patch_diag2 = patched_ids.iloc[kline]['patch_diag2']
        str_coords = pd.Series(patch_diag2).str[1:-1].str.split(', ')
        diag2 = str_coords.apply(pd.Series).astype(float)
        
        centroid_s = centroid[2]
        centoroid_y = np.min([diag1[0],diag2[0]]) + np.abs(diag1[0] - diag2[0])/2
        centoroid_x = np.min([diag1[1],diag2[1]]) + np.abs(diag1[1] - diag2[1])/2
        
        #############################
        ###### Retrieving mhas
        #############################
        slicestime = run_mha_lesion(mha_lesion_loc, str(PatientID), str(AccessionN))
        
        ## ====================                
        ## 3) Sample patch from Image
        # 3.1) convert sitk to np from int16 to uint8
        # 3.2) contrast stretch each image to include all intensities within 2nd and 98th percentils --> mapped to [0-255]
        #  # refs; http://scikit-image.org/docs/dev/auto_examples/plot_equalize.html; http://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm
        # 3.3) extract patches over all volume of size ha, wa, currently SdA supports 30x30
        ## ====================
        ha=10; wa=10; stride=10
        extrP = Patchifytest()
            
        positive_patches, label_cascadepatches = extrP.extractestPatches_cascade(slicestime, ha, wa, stride, centroid_s,centoroid_y,centoroid_x, diag1, diag2, output_folder)
        plt.savefig(output_folder+'/'+str(PatientID)+'_'+str(AccessionN)+'_'+'slice_'+str(centroid_s[0])+'.pdf')
        plt.show(block=False)
        plt.close()
        plt.close()


        all_positive_patches.extend( np.asarray( positive_patches ) )
        all_label_cascadepatches.extend(label_cascadepatches)
        
            
        # save to file all_positive_patches.pklz for patches
        fLpatches = gzip.open(output_folder+'/patches_firstSdA.pklz', 'wb')
        pickle.dump(all_positive_patches, fLpatches, protocol=pickle.HIGHEST_PROTOCOL)
        fLpatches.close()
        
        # save to file all_positive_patches.pklz for patches
        Lpatches = gzip.open(output_folder+'/all_labels_firstSdA.pklz', 'wb')
        pickle.dump(all_label_cascadepatches, Lpatches, protocol=pickle.HIGHEST_PROTOCOL)
        Lpatches.close()
        
    ###########
    ## Process Resuts
    ###########
    from patchifytest import *
    output_folder = 'plots_firstSdApatches'
    
    # read 
    with gzip.open(output_folder+'/patches_firstSdA.pklz', 'rb') as f:
        try:
            all_positive_patches = pickle.load(f, encoding='latin1')
        except:
            all_positive_patches = pickle.load(f)

    with gzip.open(output_folder+'/all_labels_firstSdA.pklz', 'rb') as f:
        try:
            all_label_cascadepatches = pickle.load(f, encoding='latin1')
        except:
            all_label_cascadepatches = pickle.load(f)
            
    
    # find some positive cascade prediction that correspond to lesion pathces
    ppatches_indx = [i for i,x in enumerate(all_label_cascadepatches) if x == 1]
    ppatches = [p for i,p in enumerate(all_positive_patches) if i in ppatches_indx]
    
    fig, ax = plt.subplots(nrows=int(np.sqrt(len(ppatches))), ncols=int(np.sqrt(len(ppatches)))+2, figsize=(12, 12))
    axes=ax.flatten()
    
    extrP = Patchifytest()
    Xtmp = np.asarray(ppatches)
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 12))
    # patch timep 0 
    imgX = Xtmp.reshape( Xtmp.shape[0], 4, 100)[:,0,:]
    ppatchesimgX = extrP.tile_images(X=imgX , img_shape=(10, 10), 
                       tile_shape=(int(np.sqrt(len(ppatches))), int(np.sqrt(len(ppatches)))+2),
                       tile_spacing=(1, 1))          
    ax[0].imshow(ppatchesimgX,  cmap=plt.cm.gray)
    # patch timep 1
    imgX = Xtmp.reshape( Xtmp.shape[0], 4, 100)[:,1,:]
    ppatchesimgX = extrP.tile_images(X=imgX , img_shape=(10, 10), 
                       tile_shape=(int(np.sqrt(len(ppatches))), int(np.sqrt(len(ppatches)))+2),
                       tile_spacing=(1, 1))          
    ax[1].imshow(ppatchesimgX,  cmap=plt.cm.gray)
    # patch timep 2
    imgX = Xtmp.reshape( Xtmp.shape[0], 4, 100)[:,2,:]
    ppatchesimgX = extrP.tile_images(X=imgX , img_shape=(10, 10), 
                       tile_shape=(int(np.sqrt(len(ppatches))), int(np.sqrt(len(ppatches)))+2),
                       tile_spacing=(1, 1))          
    ax[2].imshow(ppatchesimgX,  cmap=plt.cm.gray)
    # patch timep 2
    imgX = Xtmp.reshape( Xtmp.shape[0], 4, 100)[:,3,:]
    ppatchesimgX = extrP.tile_images(X=imgX , img_shape=(10, 10), 
                       tile_shape=(int(np.sqrt(len(ppatches))), int(np.sqrt(len(ppatches)))+2),
                       tile_spacing=(1, 1))          
    ax[3].imshow(ppatchesimgX,  cmap=plt.cm.gray)
    
    
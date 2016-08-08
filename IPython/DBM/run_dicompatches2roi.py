# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:50:00 2016

@author: DeepLearning
"""
import os, os.path
import fnmatch
import SimpleITK as sitk

import six.moves.cPickle as pickle
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
    if (len(PatientID) == 4 ): fPatientID=PatientID
    if (len(PatientID) == 3 ): fPatientID='0'+PatientID
    if (len(PatientID) == 2 ): fPatientID='00'+PatientID
    if (len(PatientID) == 1 ): fPatientID='000'+PatientID
     
    ## read Image (right now needs to use filename pattern matching)
    fpttrn = '*'+fPatientID+'_'+AccessionN+'_*'
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
    
    vol0f = '*'+fPatientID+'_'+AccessionN+'_'+str(int(Dynid))+'*'
    filen_Vol0 = filen_patt_match(mha_lesion_loc, vol0f)
    Vol0 = sitk.ReadImage(mha_lesion_loc+os.sep+filen_Vol0[0])
    
    vol1f = '*'+fPatientID+'_'+AccessionN+'_'+str(int(Dynid)+1)+'*'
    filen_Vol1 = filen_patt_match(mha_lesion_loc, vol1f)
    Vol1 = sitk.ReadImage(mha_lesion_loc+os.sep+filen_Vol1[0])
    
    vol2f = '*'+fPatientID+'_'+AccessionN+'_'+str(int(Dynid)+2)+'*'
    filen_Vol2 = filen_patt_match(mha_lesion_loc, vol2f)
    Vol2 = sitk.ReadImage(mha_lesion_loc+os.sep+filen_Vol2[0])
    
    vol3f = '*'+fPatientID+'_'+AccessionN+'_'+str(int(Dynid)+3)+'*'
    filen_Vol3 = filen_patt_match(mha_lesion_loc, vol3f)
    Vol3 = sitk.ReadImage(mha_lesion_loc+os.sep+filen_Vol3[0])
    
    vol4f = '*'+fPatientID+'_'+AccessionN+'_'+str(int(Dynid)+4)+'*'
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
                   Vol4]  
                   
    return slicestime
    

   
if __name__ == '__main__':
    
    # start by reading mha imgaes
    mha_lesion_loc='Y:/Breast/mha_lesion_labels_ground_truth/'
    patched_mris = 'view_allpatched_mris_miningBreastMRIdata.csv'
    topatch_mris = 'view_allLesions_textureUpdatedFeat.txt'

    # Open filename patched_mris
    patched_ids = pd.read_csv(patched_mris)
    
    # Open filename topatched_mris
    topatch_ids = open(topatch_mris,"r")
    topatch_ids.seek(0)
    line = topatch_ids.readline()
    line = topatch_ids.readline()
       
    # run first case of mris
    # Get the line: Study#, DicomExam#
    fileline = line.split()
    lesion_id = fileline[0]
    PatientID = fileline[1] # in case of not MRN
    print(fileline)

    # continue with patient not already patched
    AccessionN = fileline[2]
    dateID = fileline[3] #dateID = '2010-11-29' as '2010,11,29';
    side = fileline[4]
    typeEnh = fileline[5]
    centroidLoc = fileline[6:]
    print '\n%%%%%%%%%%%%%%%%%%%%%% Centroid Loc and slice = %s \n%%%%%%%%%%%%%%%%%%%%%%' % fileline[6:]
    
    #############################
    ###### Retrieving mhas
    #############################
    slicestime = run_mha_lesion(mha_lesion_loc, PatientID, AccessionN)
    
    #############################                
    ## 3) Sample patch from Image
    #############################
    ha=10; wa=10; stride=10
    extrP = Patchifytest()
    
    alltestpatches = extrP.extractestPatches(slicestime, ha, wa, stride, 'test', 'bestDBN_10x10x5.obj', 5)
    print(len(alltestpatches))
                

        
        
    ###########
    ## Process Resuts
    ###########
 

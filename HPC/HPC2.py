from nilearn import datasets
import nibabel as nib
import pandas as pd
import numpy as np
import os
from scipy.linalg import svd
from sklearn.decomposition import FastICA
import math

B = np.genfromtxt('mix.csv', delimiter=',')
B = B[1:6, 0:5]
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
#read fMRIs


skull_img = nib.load('SkullStripped_MNITemplate_2mm.nii')
skull = skull_img.get_fdata()
mask1 = (skull[:, :, :]!=0)*1
mask = skull.reshape((-1))>0
masking = nib.Nifti1Image(mask1, affine = skull_img.affine)

m = sum(mask)

A = np.diag((np.arange(15)+1)/2)
path = 'D:/HCP_PTN1200_recon2/groupICA/'
file_name = 'groupICA_3T_HCP1200_MSMAll_d15.ica/melodic_IC_sum.nii'

g_ics = nib.load("melodic_IC_sum.nii.gz")
g_ics = g_ics.get_fdata()
GICs = list()

for i in range(5):
    
    cur = g_ics[:,:,:,i]
    cur = cur.reshape((-1))
    GICs.append(cur[mask])

GICs = np.array(GICs)
sg = np.std(GICs)
X = np.matmul(B, GICs)
def get_real(seed):
    
    np.random.seed(seed)
    allx = np.zeros((15, m))
    for i in range(100):
        noise1 = np.random.multivariate_normal(np.zeros(5), np.diag(np.ones(5)*sg/100), size = m)
        noise2 = np.random.multivariate_normal(np.zeros(5), np.diag(np.ones(5)*sg/200), size = m)
        
        allX = X+np.transpose(noise1) + np.transpose(noise2) 
        allX = allX - np.mean(allX)
        allX = allX / np.std(allX)
        allx = np.concatenate([allx, allX])

    return allx
        
training0 = get_real(123)
test0 = get_real(321)    

training = training0[15:515,:].astype('float32')
test = test0[15:515,:].astype('float32')

from nilearn import datasets
import nibabel as nib
import pandas as pd
import numpy as np
import os
from scipy.linalg import svd
from sklearn.decomposition import FastICA
#read fMRIs

fmri_path = 'D:/pre-processed_fMRI_All/fMRI_Reg_Completed'
file_names = os.listdir(fmri_path)
image_names = dict()
for i in file_names:
    cpath = ''.join([fmri_path, '/', i])
    cpath = ''.join([cpath, '/', os.listdir(cpath)[0]])
    cpath = ''.join([cpath, '/', os.listdir(cpath)[0]])
    cpath = ''.join([cpath, '/', os.listdir(cpath)[0]])
    sub_files = os.listdir(cpath)
    sub_names = list()
    for j in range(len(sub_files)):
        spath = ''.join([cpath, '/', sub_files[j]])
        sub_names.append(spath)
    image_names[i] = sub_names

mask = cur>0

fmris = dict()

for ids in file_names[20:58]:
    print(ids)
    niis = image_names[ids]
    num = len(niis)
    if(num == 197):
        X = list()
        for i in range(num):
            print(i)
            cur = nib.load(niis[i])
            cur = cur.get_fdata()
            cur = cur.reshape((-1))
            X.append(cur[mask])
        fmris[ids] = X
            
#construct fmris
svd_fmris = list()
for key in fmris.keys():
    X = np.array(fmris[key])
    XX = np.matmul(X, np.transpose(X))
    U, singular, V_transpose = svd(XX)
    U_t = np.transpose(U)
    svd_fmris = svd_fmris + list(np.matmul(U_t[0:20,],X))
    
all_fmris = np.array(svd_fmris)
XX =  np.matmul(all_fmris, np.transpose(all_fmris))
U, singular, V_transpose = svd(XX)
U_t = np.transpose(U)

# filenames = pd.read_csv('file.csv')
# files = list(filenames['x'])
# background = nib.load('ADNI_024_S_6202_MR_Axial_rsfMRI__Eyes_Open__Phase_Direction_P_A_br_raw_20180213173615786_151_S658703_I963560.nii_registered.nii.gz')
tmp = nib.load(image_names[file_names[0]][0])
t1 = tmp.get_fdata()
mask1 = (t1[:, :, :]!=0)*1
masking = nib.Nifti1Image(mask1, affine = tmp.affine)
#fastICA for fmri

    
from nilearn.maskers import NiftiMasker
masker = NiftiMasker(mask_img = masking)
data_masked = masker.fit_transform(tmp)

    

n_components = 15
ica = FastICA(n_components=n_components, random_state=42)
components_masked = ica.fit_transform(all_fmris.T).T
components_masked -= components_masked.mean(axis=0)
components_masked /= components_masked.std(axis=0)   
components_masked[np.abs(components_masked) < 0.8] = 0

# Now invert the masking operation, going back to a full 3D
# representation
component_img = masker.inverse_transform(components_masked)
from nilearn.plotting import plot_prob_atlas
plot_prob_atlas(component_img)
nib.save(component_img, 'fmris.nii')
#read group ICs (HCP)
gics = nib.load('melodic_IC_sum.nii.gz')
from nilearn.plotting import plot_prob_atlas
plot_prob_atlas(gics)


#ICs estimated from templateICA(TAU+fMRIs)
ad = nib.load('ad.nii.gz')
ad = nib.Nifti1Image(ad.get_fdata(), affine = component_img.affine)
plot_prob_atlas(ad)

mci = nib.load('mci.nii.gz')
mci = nib.Nifti1Image(mci.get_fdata(), affine = component_img.affine)
plot_prob_atlas(mci)

nc = nib.load('nc.nii.gz')
nc = nib.Nifti1Image(nc.get_fdata(), affine = component_img.affine)
plot_prob_atlas(nc)


#read TAU PET
tau_path = 'Code/Info/TAU/tau.csv'
tau_file = pd.read_csv(tau_path)

cur_file = tau_file['files'][0]
cur_img = nib.load(cur_file)
cur_tau = cur_img.get_fdata()
tau_mask = (cur_tau[:, :, :]!=0)*1
vec_mask = cur_tau.reshape((-1))>0
tau_mask = nib.Nifti1Image(tau_mask, affine = cur_img.affine)


all_tau = list()
for i in range(len(tau_file)):
    
    cur_file = tau_file['files'][i]
    cur_img = nib.load(cur_file)
    cur_tau = cur_img.get_fdata()
    cur_tau = cur_tau.reshape((-1))
    all_tau.append(cur_tau[vec_mask])
    
all_tau = np.array(all_tau)

#masking
from nilearn.maskers import NiftiMasker
masker = NiftiMasker(mask_img = tau_mask)
tau_masked = masker.fit_transform(cur_img)
#fastICA
ica = FastICA(n_components=n_components, random_state=42)
components_masked = ica.fit_transform(all_tau.T).T
components_masked -= components_masked.mean(axis=0)
components_masked /= components_masked.std(axis=0)   
components_masked[np.abs(components_masked) < 0.8] = 0
    
# Now invert the masking operation, going back to a full 3D
# representation
component_img = masker.inverse_transform(components_masked)

from nilearn.plotting import plot_prob_atlas
plot_prob_atlas(component_img)
nib.save(component_img, 'all_tau.nii')

all_dx = tau_file['DXCURREN'].tolist()
nc_ind = [i for i,x in enumerate(all_dx) if x==1]
mci_ind = [i for i,x in enumerate(all_dx) if x==2]
ad_ind = [i for i,x in enumerate(all_dx) if x==3]


#NC
nc_tau = list()
for i in nc_ind:
    cur_file = tau_file['files'][i]
    cur_img = nib.load(cur_file)
    cur_tau = cur_img.get_fdata()
    cur_tau = cur_tau.reshape((-1))
    nc_tau.append(cur_tau[vec_mask])
    
nc_tau = np.array(nc_tau)

from nilearn.maskers import NiftiMasker
masker = NiftiMasker(mask_img = tau_mask)
tau_masked = masker.fit_transform(cur_img)
#fastICA
ica = FastICA(n_components=n_components, random_state=42)
components_masked = ica.fit_transform(nc_tau.T).T
components_masked -= components_masked.mean(axis=0)
components_masked /= components_masked.std(axis=0)   
components_masked[np.abs(components_masked) < 0.8] = 0
    
# Now invert the masking operation, going back to a full 3D
# representation
component_img = masker.inverse_transform(components_masked)

from nilearn.plotting import plot_prob_atlas
plot_prob_atlas(component_img)
nib.save(component_img, 'nc_tau.nii')

#MCI
mci_tau = list()
for i in mci_ind:
    cur_file = tau_file['files'][i]
    cur_img = nib.load(cur_file)
    cur_tau = cur_img.get_fdata()
    cur_tau = cur_tau.reshape((-1))
    mci_tau.append(cur_tau[vec_mask])
    
mci_tau = np.array(mci_tau)

from nilearn.maskers import NiftiMasker
masker = NiftiMasker(mask_img = tau_mask)
tau_masked = masker.fit_transform(cur_img)
#fastICA
ica = FastICA(n_components=n_components, random_state=42)
components_masked = ica.fit_transform(mci_tau.T).T
components_masked -= components_masked.mean(axis=0)
components_masked /= components_masked.std(axis=0)   
components_masked[np.abs(components_masked) < 0.8] = 0
    
# Now invert the masking operation, going back to a full 3D
# representation
component_img = masker.inverse_transform(components_masked)

from nilearn.plotting import plot_prob_atlas
plot_prob_atlas(component_img)
nib.save(component_img, 'mci_tau.nii')


#AD
ad_tau = list()
for i in ad_ind:
    cur_file = tau_file['files'][i]
    cur_img = nib.load(cur_file)
    cur_tau = cur_img.get_fdata()
    cur_tau = cur_tau.reshape((-1))
    ad_tau.append(cur_tau[vec_mask])
    
ad_tau = np.array(ad_tau)

from nilearn.maskers import NiftiMasker
masker = NiftiMasker(mask_img = tau_mask)
tau_masked = masker.fit_transform(cur_img)
#fastICA
ica = FastICA(n_components=n_components, random_state=42)
components_masked = ica.fit_transform(ad_tau.T).T
components_masked -= components_masked.mean(axis=0)
components_masked /= components_masked.std(axis=0)   
components_masked[np.abs(components_masked) < 0.8] = 0
    
# Now invert the masking operation, going back to a full 3D
# representation
component_img = masker.inverse_transform(components_masked)

from nilearn.plotting import plot_prob_atlas
plot_prob_atlas(component_img)
nib.save(component_img, 'ad_tau.nii')

dmn = nib.load('dmn.nii.gz')
from nilearn import image
for i in range(4):
    plot_stat_map(image.index_img(dmn, i), backg)
from nilearn.plotting import plot_stat_map, show
mean_img = image.mean_img('MNI152_T1_2mm.nii')
plot_stat_map(image.index_img(gics, 0))

plot_stat_map(image.index_img(component_img, 1))

show()
from nilearn import datasets
import nibabel as nib
import pandas as pd
import numpy as np
import os
from scipy.linalg import svd
from sklearn.decomposition import FastICA
from nilearn import image

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


skull_img = nib.load('SkullStripped_MNITemplate_2mm.nii')
skull = skull_img.get_fdata()
mask1 = (skull[:, :, :]!=0)*1
mask = skull.reshape((-1))>0
masking = nib.Nifti1Image(mask1, affine = skull_img.affine)

nc_fmri = pd.read_csv('nc_fmri.csv')
mci_fmri = pd.read_csv('mci_fmri.csv')
ad_fmri = pd.read_csv('ad_fmri.csv')

fmris = dict()
nc_fmris = dict()
mci_fmris = dict()
ad_fmris = dict()

def get_img(dicts):
    ret_dict = dict()
    for ids in dicts['PTID'].tolist():
    
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
            ret_dict[ids] = X
    return ret_dict

nc_fmris = get_img(nc_fmri)
mci_fmris = get_img(mci_fmri)
ad_fmris = get_img(ad_fmri)

fmris = {**nc_fmris, **mci_fmris, **ad_fmris}


#construct fmris
def get_ic(dicts):
    
    svd_fmris = list()
    for key in dicts.keys():
        X = np.array(dicts[key])
        XX = np.matmul(X, np.transpose(X))
        U, singular, V_transpose = svd(XX)
        U_t = np.transpose(U)
        svd_fmris = svd_fmris + list(np.matmul(U_t[0:30,],X))

    all_fmris = np.array(svd_fmris)
    return all_fmris

nc_svd = get_ic(nc_fmris)
mci_svd = get_ic(mci_fmris)
ad_svd = get_ic(ad_fmris)
all_svd = get_ic(fmris)


# XX = np.matmul(all_fmris, np.transpose(all_fmris))
# U, singular, V_transpose = svd(XX)
# U_t = np.transpose(U)

# filenames = pd.read_csv('file.csv')
# files = list(filenames['x'])
# background = nib.load('ADNI_024_S_6202_MR_Axial_rsfMRI__Eyes_Open__Phase_Direction_P_A_br_raw_20180213173615786_151_S658703_I963560.nii_registered.nii.gz')

#fastICA for fmri
fs = all_svd
def visu_ic(fs):
    
    from nilearn.maskers import NiftiMasker
    masker = NiftiMasker(mask_img = masking)
    data_masked = masker.fit_transform(skull_img)

        
    n_components = 25
    ica = FastICA(n_components=n_components, random_state=42)
    components_masked = ica.fit_transform(fs.T).T
    components_masked -= components_masked.mean(axis=0)
    components_masked /= components_masked.std(axis=0)   
    components_masked[np.abs(components_masked) < 0.8] = 0

    # Now invert the masking operation, going back to a full 3D
    # representation
    component_img = masker.inverse_transform(components_masked)
    return component_img

nc_img = visu_ic(nc_svd)
mci_img = visu_ic(mci_svd)
ad_img = visu_ic(ad_svd)
all_img = visu_ic(all_svd)

all_img = all_img.get_fdata()
all_fmri = all_img
for i in range(25):
    all_fmri[:,:,:,i] = all_img[:,:,:,i]*mask1
    
all_fmri = nib.Nifti1Image(all_fmri, affine = skull_img.affine)

from nilearn.plotting import plot_prob_atlas
plot_prob_atlas(nc_img, bg_img = skull_img, cut_coords = (0,4,22), view_type = 'filled_contours')
plot_prob_atlas(mci_img, bg_img = skull_img, cut_coords = (0,4,22), view_type = 'filled_contours')
plot_prob_atlas(ad_img, bg_img = skull_img, cut_coords = (0,4,22), view_type = 'filled_contours')
plot_prob_atlas(all_img, bg_img = skull_img, cut_coords = (0,4,22), view_type = 'filled_contours')

nib.save(all_img, 'IC25/all_fmri.nii')
nib.save(nc_img, 'IC25/nc_fmri.nii')
nib.save(mci_img, 'IC25/mci_fmri.nii')
nib.save(ad_img, 'IC25/ad_fmri.nii')
#read group ICs (HCP)

#GIFT

gift_fmri = nib.load('fmris_mean_component_ica_s1_.nii')
tmp = gift_fmri.get_fdata()
tmp[np.abs(tmp)<0.5]=0
gift_fmri = nib.Nifti1Image(tmp, affine = skull_img.affine)
from nilearn.plotting import plot_prob_atlas
plot_prob_atlas(gift_fmri, cut_coords = (0,4,22), bg_img = skull_img)

gift_tau = nib.load('_sub01_component_ica_s1_all.nii')
from nilearn.plotting import plot_prob_atlas
plot_prob_atlas(gift_tau, cut_coords = (0,4,22), bg_img = skull_img)

gift_nctau = nib.load('nc_sub01_component_ica_s1_.nii')
from nilearn.plotting import plot_prob_atlas
plot_prob_atlas(gift_nctau, cut_coords = (0,4,22), bg_img = skull_img)

gift_mcitau = nib.load('mci_sub01_component_ica_s1_.nii')
from nilearn.plotting import plot_prob_atlas
plot_prob_atlas(gift_mcitau, cut_coords = (0,4,22), bg_img = skull_img)

gift_adtau = nib.load('ad_sub01_component_ica_s1_.nii')
from nilearn.plotting import plot_prob_atlas
plot_prob_atlas(gift_adtau, cut_coords = (0,4,22), bg_img = skull_img)


gics = nib.load('IC25/melodic_IC_sum.nii.gz')
from nilearn.plotting import plot_prob_atlas
plot_prob_atlas(gics, bg_img = skull_img, cut_coords = (0,4,22), view_type = 'filled_contours')


#ICs estimated from templateICA(TAU+fMRIs)
ad = nib.load('IC25/ad08.nii.gz')
ad = nib.Nifti1Image(ad.get_fdata(), affine = skull_img.affine)
plot_prob_atlas(ad, bg_img = skull_img, cut_coords = (0,4,22), view_type = 'filled_contours')

mci = nib.load('IC25/mci08.nii.gz')
mci = nib.Nifti1Image(mci.get_fdata(), affine = skull_img.affine)
plot_prob_atlas(mci, bg_img = skull_img, cut_coords = (0,10,22), view_type = 'filled_contours')

nc = nib.load('IC25/nc08.nii.gz')
nc = nib.Nifti1Image(nc.get_fdata(), affine = skull_img.affine)
plot_prob_atlas(nc, bg_img = skull_img, cut_coords = (0,4,22), view_type = 'filled_contours')

alls = nib.load('IC25/all08.nii.gz')
plot_prob_atlas(alls, bg_img = skull_img, cut_coords = (0,4,22), view_type = 'filled_contours')


#read TAU PET
tau_path = 'D:/Yimo/PhD/Fall 2020/RA/Code/Info/TAU/taus.csv'
tau_file = pd.read_csv(tau_path)


skull_img = nib.load('SkullStripped_MNITemplate_2mm.nii')
skull = skull_img.get_fdata()
tau_mask = (skull[:, :, :]!=0)*1
vec_mask = skull.reshape((-1))>0
tau_mask = nib.Nifti1Image(tau_mask, affine = skull_img.affine)


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
tau_masked = masker.fit_transform(skull_img)
#fastICA
ica = FastICA(n_components=25, random_state=42)
components_masked = ica.fit_transform(all_tau.T).T
components_masked -= components_masked.mean(axis=0)
components_masked /= components_masked.std(axis=0)   
components_masked[np.abs(components_masked) < 0.8] = 0
    
# Now invert the masking operation, going back to a full 3D
# representation
component_img = masker.inverse_transform(components_masked)
tt = component_img.get_fdata()
from nilearn.plotting import plot_prob_atlas
plot_prob_atlas(component_img, bg_img = skull_img, cut_coords = (0,4,22), view_type = 'filled_contours')
nib.save(component_img, 'IC25/all_tau.nii')



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
tau_masked = masker.fit_transform(skull_img)
#fastICA
ica = FastICA(n_components=25, random_state=42)
components_masked = ica.fit_transform(nc_tau.T).T
components_masked -= components_masked.mean(axis=0)
components_masked /= components_masked.std(axis=0)   
components_masked[np.abs(components_masked) < 0.8] = 0
    
# Now invert the masking operation, going back to a full 3D
# representation
component_img = masker.inverse_transform(components_masked)

from nilearn.plotting import plot_prob_atlas
plot_prob_atlas(component_img, bg_img = skull_img, cut_coords = (0,4,22), view_type = 'filled_contours')
nib.save(component_img, 'IC25/nc_tau.nii')

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
tau_masked = masker.fit_transform(skull_img)
#fastICA
ica = FastICA(n_components=25, random_state=42)
components_masked = ica.fit_transform(mci_tau.T).T
components_masked -= components_masked.mean(axis=0)
components_masked /= components_masked.std(axis=0)   
components_masked[np.abs(components_masked) < 0.8] = 0
    
# Now invert the masking operation, going back to a full 3D
# representation
component_img = masker.inverse_transform(components_masked)

from nilearn.plotting import plot_prob_atlas
plot_prob_atlas(gift_fmri, bg_img = skull_img, cut_coords = (0,4,22), view_type = 'filled_contours')
nib.save(component_img, 'IC25/mci_tau.nii')


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
tau_masked = masker.fit_transform(skull_img)
#fastICA
ica = FastICA(n_components=25, random_state=42)
components_masked = ica.fit_transform(ad_tau.T).T
components_masked -= components_masked.mean(axis=0)
components_masked /= components_masked.std(axis=0)   
components_masked[np.abs(components_masked) < 0.8] = 0
    
# Now invert the masking operation, going back to a full 3D
# representation
component_img = masker.inverse_transform(components_masked)

from nilearn.plotting import plot_prob_atlas
plot_prob_atlas(component_img, bg_img = skull_img, cut_coords = (0,4,22), view_type = 'filled_contours')
nib.save(component_img, 'IC25/ad_tau.nii')

from nilearn.plotting import plot_prob_atlas
plot_prob_atlas(component_img, bg_img = skull_img, cut_coords = (0,4,22), view_type = 'filled_contours')
nib.save(component_img, 'IC25/ad_tau.nii')

from nilearn.plotting import plot_stat_map, show


fmri_over = nib.load('IC25/all_fmri.nii')

tt = fmri_over.get_fdata()
tau_over = nib.load('IC25/all_tau.nii')
temp_over = nib.load('IC25/all.nii.gz')

gift_fmri = nib.load('IC25/gift_fmri.nii')
gift_fmri_data = gift_fmri.get_fdata()
gift_fmri = nib.Nifti1Image(gift_fmri_data, affine = skull_img.affine)


tau = nib.load('IC25/all_tau.nii')
indx = [2,17,18]
for i in indx:
    plot_stat_map(image.index_img(tau, i-1), bg_img = skull_img, cut_coords = (0,4,22))

fmri = nib.load('IC25/all_fmri.nii')
indx = [6,24]
for i in indx:
    plot_stat_map(image.index_img(fmri, i-1), bg_img = skull_img, cut_coords = (0,4,22))

temp = nib.load('IC25/all.nii.gz')
indx=[1,3,5,6,8,10,11,14,19,21]
for i in indx:
    plot_stat_map(image.index_img(temp, i-1), bg_img = skull_img, cut_coords = (0,4,22))


plot_prob_atlas(dmn, bg_img = skull_img, cut_coords = (0,4,22), view_type = 'filled_contours')
gift_fmri_mask = nib.load('IC25/fmri25Mask.nii')

aal = nib.load('aal2.nii.gz')
dmn = nib.load('dmn.nii.gz')
plot_prob_atlas(dmn)
for i in range(4):
    plot_stat_map(image.index_img(dmn, i), bg_img = skull_img, vmax = 45)
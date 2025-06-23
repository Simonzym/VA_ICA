library(fslr)
library(RNifti)
library(templateICAr)
library(stringr)
library(dplyr)
library(fmri)
library(fastICA)


#group ICs
path = 'D:/HCP_PTN1200_recon2/groupICA'
file.name = 'groupICA_3T_HCP1200_MSMAll_d25.ica/melodic_IC_sum.nii'
g_ics = readNIfTI(paste0(path, '/', file.name))
GICs = c()
setwd('Y:/PhD/Fall 2020/RA/')
fic = readNIfTI('fic.nii.gz')
for(i in 1:25){
  cur = as.vector(g_ics[,,,i])
  GICs = rbind(GICs, cur[mask])
}
GICs = t(GICs)

#fMRIs
fmri_path = 'D:/pre-processed_fMRI_All/fMRI_Reg_Completed'

file_names = dir(fmri_path)
image_names = list()

for(i in file_names){
  cpath = paste0(fmri_path, '/', i)
  cpath = paste0(cpath, '/', dir(cpath)[1])
  cpath = paste0(cpath, '/', dir(cpath)[1])
  cpath = paste0(cpath, '/', dir(cpath)[1])
  sub_files = dir(cpath)
  sub_names = c()
  for(j in 1:length(sub_files)){
    spath = paste0(cpath, '/', sub_files[j])
    sub_names = c(sub_names, spath)
  }
  image_names[[i]] = sub_names
}

ids = names(image_names)
fmris = list()
#mask = cur>0
for(name in ids[1:5]){
  niis = image_names[[name]]
  num = length(niis)
  X = c()
  for(i in 1:num){
    cur = readNIfTI(niis[i])
    cur = as.vector(cur)
    cur = cur[mask]
    X = rbind(X, cur)
  }
  print(name)
  fmris[[name]] = t(X)
}

#transpose

fmri = readRDS('D:/fmris/fmris.RData')
for(i in 1:57){
  
  tmp = dim(fmri[[i]])
  if(tmp[2]!= 197){print(i)}
}

fmris = fmri[-c(11,55)]
f1 = fmris[1:5]


#TAU-PET
setwd('Y:/PhD/Fall 2020/RA/')
tau = read.csv('Code/Info/TAU/tau.csv')

normal = c()
mci = c()
ad = c()

normal_data = tau%>%filter(DXCURREN == 1)
mci_data = tau%>%filter(DXCURREN == 2)
ad_data = tau%>%filter(DXCURREN == 3)

for(i in 1:dim(normal_data)[1]){
  item = normal_data[i,]
  file.name = str_replace(item$files, 'TAU9_3', 'TAU9')
  file.name = str_replace(file.name, 'sub1', 'aal')
  cur = readNIfTI(file.name)
  cur = as.vector(cur)
  normal = rbind(normal, cur[mask])
}

for(i in 1:dim(mci_data)[1]){
  item = mci_data[i,]
  file.name = str_replace(item$files, 'TAU9_3', 'TAU9')
  file.name = str_replace(file.name, 'sub1', 'aal')
  cur = readNIfTI(file.name)
  cur = as.vector(cur)
  mci = rbind(mci, cur[mask])
}

for(i in 1:dim(ad_data)[1]){
  item = ad_data[i,]
  file.name = str_replace(item$files, 'TAU9_3', 'TAU9')
  file.name = str_replace(file.name, 'sub1', 'aal')
  cur = readNIfTI(file.name)
  cur = as.vector(cur)
  ad = rbind(ad, cur[mask])
}

#estimate template
nQ = dim(GICs)[2]
nV = dim(GICs)[1]
nT = 197
mS = GICs%*%diag(seq(nQ,1)) %*%matrix(rnorm(nQ*nT),nrow=nQ)
BOLD = list(B1=mS,B2=mS,B3=mS)
BOLD = lapply(BOLD,function(x){x+rnorm(nV*nT,sd=.05)})

tmps = estimate_template(BOLD = fmris, GICA=GICs, maskTol = 0.99, varTol = 1e-20, missingTol = 0.99)

# GICs = GICs-mean(GICs)
# xx = solve(t(GICs)%*%GICs)%*%t(GICs)
# ss = list()
# 
# for(i in 1:55){
#   name = names(fmris)[i]
#   X = fmris[[name]]
#   M = xx%*%X
#   print(det(M%*%t(M)))
# }

normal_result = templateICA(t(normal), tmps, Q2 = 5,  varTol = 0)
mci_result = templateICA(t(mci), tmps, Q2 = 5,  varTol = 0)
ad_result = templateICA(t(ad), tmps,  Q2= 5,varTol = 0)

all_result = templateICA(t(rbind(normal, mci, ad)), Q2 = 5, tmps, varTol = 0)

tem_nc = normal_result$A
tem_mci = mci_result$A
tem_ad = ad_result$A

labels =  c('NC', 'MCI', 'AD')
par(mfrow=c(3,5))
for(i in 1:15){
  boxplot(tem_nc[,i], tem_mci[,i], tem_ad[,i], names = labels)
}


nc_mean = normal_result$subjICmean
mci_mean = mci_result$subjICmean
ad_mean = ad_result$subjICmean

nc_mean[is.na(nc_mean)] = 0
mci_mean[is.na(mci_mean)] = 0
ad_mean[is.na(ad_mean)] = 0

nc_mean = nc_mean - mean(nc_mean)
mci_mean = mci_mean - mean(mci_mean)
ad_mean = ad_mean - mean(ad_mean)

nc_mean = nc_mean/sd(nc_mean)
mci_mean = mci_mean/sd(mci_mean)
ad_mean = ad_mean/sd(ad_mean)

nc_mean[abs(nc_mean)<0.8] = 0
mci_mean[abs(mci_mean)<0.8] = 0
ad_mean[abs(ad_mean)<0.8] = 0

ncs= matrix(0, nrow = 91*109*91, ncol = 15)
mcis = matrix(0, nrow = 91*109*91, ncol = 15)
ads = matrix(0, nrow = 91*109*91, ncol = 15)

for(i in 1:15){
  
  ncs[mask,i] = nc_mean[,i]
  mcis[mask, i] = mci_mean[,i]
  ads[mask, i] = ad_mean[,i]
}

ncs = array(ncs, dim = c(91, 109, 91, 15))
mcis = array(mcis, dim = c(91, 109, 91, 15))
ads = array(ads, dim = c(91, 109, 91, 15))

ncs = as.nifti(ncs)
mcis = as.nifti(mcis)
ads = as.nifti(ads)

writeNIfTI(ncs, 'nc')
writeNIfTI(mcis, 'mci')
writeNIfTI(ads, 'ad')

normal = c()
mci = c()
ad = c()

normal_data = tau%>%filter(DXCURREN == 1)
mci_data = tau%>%filter(DXCURREN == 2)
ad_data = tau%>%filter(DXCURREN == 3)

for(i in 1:dim(normal_data)[1]){
  item = normal_data[i,]
  cur = readNIfTI(item$files)
  normal = rbind(normal, as.vector(cur))
}

for(i in 1:dim(mci_data)[1]){
  item = mci_data[i,]
  cur = readNIfTI(item$files)
  mci = rbind(mci, cur)
}

for(i in 1:dim(ad_data)[1]){
  item = ad_data[i,]
  cur = readNIfTI(item$files)
  ad = rbind(ad, cur)
}


std_gics = GICs-mean(GICs)
std_gics = std_gics/sd(std_gics)
std_gics[abs(std_gics)<0.8] = 0

par(mfrow=c(2,5))
for(i in 1:10){
  cur_nc = array(std_gics[,15], dim = c(91,109,91))
  #cur_nc = array(ad_mean[,15], dim = c(91, 109, 91))
  #cur_mci = array(mci_mean[,i], dim = c(91, 109, 91))
  #cur_ad = array(ad_mean[,i], dim = c(91, 109, 91))
  orthographic(cur_nc, col = gray(0:64/64))
}

#group ICA in fMRIs
setwd('D:/fmri')
for(i in 1:55){
  name = names(fmri)[i]
  X = fmri[[name]]
  file_name = paste0('fmri', i)
  writeNIfTI(X, file_name)
}


library(fslr)
library(RNifti)
library(templateICAr)
library(stringr)
library(dplyr)
library(fastICA)


#group ICs
path = 'D:/HCP_PTN1200_recon2/groupICA'
file.name = 'groupICA_3T_HCP1200_MSMAll_d25.ica/melodic_IC_sum.nii'
g_ics = readNIfTI(paste0(path, '/', file.name))
GICs = c()
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
setwd('D:/Yimo/PhD/Thesis3')
ids = names(image_names)
skull = readNIfTI('SkullStripped_MNITemplate_2mm.nii')
fmris = list()
mask = skull>0
writeNIfTI(mask, 'fmri25Mask', gzipped = F)

for(name in ids[-19]){
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

saveRDS(fmris, "fmris25.RData")

fmris_name = tau%>%filter(PTID %in% names(image_names)[-19])
nc_fmri = fmris_name%>%filter(DXCURREN==1)
mci_fmri = fmris_name%>%filter(DXCURREN==2)
ad_fmri = fmris_name%>%filter(DXCURREN==3)
setwd("Y:/PhD/Thesis3")
write.csv(nc_fmri, 'nc_fmri.csv')
write.csv(mci_fmri, 'mci_fmri.csv')
write.csv(ad_fmri, 'ad_fmri.csv')
#TAU-PET
setwd('Y:/PhD/Fall 2020/RA/')
tau = read.csv('Code/Info/TAU/tau.csv')


normal = c()
mci = c()
ad = c()

taus = tau
taus$files = str_replace(taus$files, 'Y:', 'D:/Yimo')
taus$files = str_replace(taus$files, 'TAU9_3', 'TAU9')
taus$files = str_replace(taus$files, 'sub1', 'aal')
write.csv(taus, 'Code/Info/TAU/taus.csv')
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

tmps = estimate_template(BOLD = fmris, GICA=GICs,  missingTol = 0.4, maskTol = 0.4)

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

normal_result = templateICA(t(normal), tmps, Q2 = 0,  varTol = 1e-20)
mci_result = templateICA(t(mci), tmps, Q2 = 0,  varTol = 1e-20)
ad_result = templateICA(t(ad), tmps,  Q2= 0,varTol = 1e-20)
tmps = readRDS('Y:/PhD/Thesis3/IC25/tmps.rds')
all_result = templateICA(t(rbind(normal, mci, ad)), Q2 = 0, tmps, varTol = 1e-20)

tem_nc = normal_result$A
tem_mci = mci_result$A
tem_ad = ad_result$A

saveRDS(tmps, 'Y:/PhD/Thesis3/IC25/tmps.rds')
saveRDS(normal_result, 'Y:/PhD/Thesis3/IC25/nc.rds')
saveRDS(mci_result, 'Y:/PhD/Thesis3/IC25/mci.rds')
saveRDS(ad_result, 'Y:/PhD/Thesis3/IC25/ad.rds')
saveRDS(all_result, 'Y:/PhD/Thesis3/IC25/all.rds')

all_result = readRDS('Y:/PhD/Thesis3/IC25/all.rds')
mci_result = readRDS('Y:/PhD/Thesis3/IC25/mci.rds')
ad_result = readRDS('Y:/PhD/Thesis3/IC25/ad.rds')
normal_result = readRDS('Y:/PhD/Thesis3/IC25/nc.rds')

labels =  c('NC', 'MCI', 'AD')
par(mfrow=c(1,3), mar = c(2, 2, 2,2))
for(i in c(4,8,19)){
  boxplot(tem_nc[,i], tem_mci[,i], tem_ad[,i], names = labels, main = paste0('IC', i))
}


nc_mean = normal_result$subjICmean
mci_mean = mci_result$subjICmean
ad_mean = ad_result$subjICmean
all_mean = all_result$subjICmean

trans_back = function(ic_mean, thre = 0.8){
  
  ic_mean = ic_mean - apply(ic_mean, 1, function(x) mean(x, na.rm = T))
  ic_mean = ic_mean/apply(ic_mean, 1, function(x) sd(x, na.rm = T))
  ic_mean[abs(ic_mean)<thre] = 0
  ics = array(0, dim = c(91, 109, 91, 25))
  for(i in 1:25){
    cur.ic = t(ic_mean)[i,]
    cur.img = array(0, dim = c(91, 109, 91))
    cur.img[mask] = cur.ic
    ics[,,,i] = cur.img
    
  }
  ics[is.na(ics)] = 0
  return(ics)
}

ncs = trans_back(nc_mean, 2)
mcis = trans_back(mci_mean, 2)
ads = trans_back(ad_mean, 2)
alls = trans_back(all_mean, 2)

ncs_out = g_ics
mcis_out = g_ics
ads_out = g_ics
alls_out = g_ics

ncs_out@.Data = ncs
mcis_out@.Data = mcis
ads_out@.Data = ads
alls_out@.Data = alls

writeNIfTI(ncs_out, 'Y:/PhD/Thesis3/IC25/nc2')
writeNIfTI(mcis_out, 'Y:/PhD/Thesis3/IC25/mci2')
writeNIfTI(ads_out, 'Y:/PhD/Thesis3/IC25/ad2')
writeNIfTI(alls_out, 'Y:/PhD/Thesis3/IC25/all2')

# normal = c()
# mci = c()
# ad = c()
# 
# normal_data = tau%>%filter(DXCURREN == 1)
# mci_data = tau%>%filter(DXCURREN == 2)
# ad_data = tau%>%filter(DXCURREN == 3)
# 
# for(i in 1:dim(normal_data)[1]){
#   item = normal_data[i,]
#   cur = readNIfTI(item$files)
#   normal = rbind(normal, as.vector(cur))
# }
# 
# for(i in 1:dim(mci_data)[1]){
#   item = mci_data[i,]
#   cur = readNIfTI(item$files)
#   mci = rbind(mci, cur)
# }
# 
# for(i in 1:dim(ad_data)[1]){
#   item = ad_data[i,]
#   cur = readNIfTI(item$files)
#   ad = rbind(ad, cur)
# }
# 
# 
# std_gics = GICs-mean(GICs)
# std_gics = std_gics/sd(std_gics)
# std_gics[abs(std_gics)<0.8] = 0
# 
# par(mfrow=c(2,5))
# for(i in 1:10){
#   cur_nc = array(std_gics[,15], dim = c(91,109,91))
#   #cur_nc = array(ad_mean[,15], dim = c(91, 109, 91))
#   #cur_mci = array(mci_mean[,i], dim = c(91, 109, 91))
#   #cur_ad = array(ad_mean[,i], dim = c(91, 109, 91))
#   orthographic(cur_nc, col = gray(0:64/64))
# }
# 
# #group ICA in fMRIs
# setwd('D:/fmri')
# for(i in 1:55){
#   name = names(fmri)[i]
#   X = fmri[[name]]
#   file_name = paste0('fmri', i)
#   writeNIfTI(X, file_name)
# }


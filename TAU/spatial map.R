library(fslr)
library(RNifti)
library(templateICAr)
library(stringr)
library(dplyr)
library(fastICA)
library(corrplot)
library(R.matlab)

setwd('Y:/PhD/Thesis3/IC25')

#read VA_ICA TAU space map
all.vaica = readNifti('vaica_sub01_component_ica_s1_.nii')

#read fMRI space map
all.fmri = readNIfTI('fmris.nii')
nc.fmri = readNIfTI('nc_fmri.nii')
mci.fmri = readNIfTI('mci_fmri.nii')
ad.fmri = readNIfTI('ad_fmri.nii')

#read TAU-only space map
all.tau = readNIfTI('all_tau.nii')
nc.tau = readNIfTI('nc_tau.nii')
mci.tau = readNIfTI('mci_tau.nii')
ad.tau = readNIfTI('ad_tau.nii')

#read TAU template
all.temp = readNIfTI('all.nii')
nc.temp = readNIfTI('nc.nii')
mci.temp = readNIfTI('mci.nii')
ad.temp = readNIfTI('ad.nii')

skull = readNIfTI('SkullStripped_MNITemplate_2mm.nii')
skull_vec = as.vector(skull)
fmri_vec = as.vector(all.fmri[,,,6])
all.temp = rid_na(all.temp)
rid_na = function(x){
  x[is.na(x)]=0
  return(x)
}

trans_array = function(img, num=25){
  mt = array(0, dim = c(num, 91*109*91))
  for(i in 1:num){
    mt[i,] = as.vector(img[,,,i])
  }
  mt[is.na(mt)] = 0
  return(mt)
}

mc = readNIfTI('melodic_IC_sum.nii')
mc = trans_array(mc)
mc = trans_back(t(mc))
trans_back = function(ic_mean){
  
  ic_mean = ic_mean - apply(ic_mean, 1, function(x) mean(x, na.rm = T))
  ic_mean = ic_mean/apply(ic_mean, 1, function(x) sd(x, na.rm = T))
  ic_mean[abs(ic_mean)<0.8] = 0
  ics = array(0, dim = c(91, 109, 91, 25))
  for(i in 1:25){
    cur.ic = t(ic_mean)[i,]
    ics[,,,i] = cur.ic
    
  }
  ics[is.na(ics)] = 0
  return(ics)
}

all.fmri = trans_array(all.fmri)
nc.fmri = trans_array(nc.fmri)
mci.fmri = trans_array(mci.fmri)
ad.fmri = trans_array(ad.fmri)

all.tau = trans_array(all.tau)
ad.tau = trans_array(ad.tau)
mci.tau = trans_array(mci.tau)
ad.tau = trans_array(ad.tau)

all.temp = trans_array(all.temp)
nc.temp = trans_array(nc.temp)
mci.temp = trans_array(mci.temp)
ad.temp = trans_array(ad.temp)

mc = trans_array(mc)

fmri.cor = cor(t(all.fmri), method = 'pearson')
tau.cor = cor(t(all.tau), method = 'pearson')
temp.cor = cor(t(all.temp), method = 'pearson')
mc.cor = cor(t(mc), method = 'pearson')


fmri.tau.cor = cor(t(all.fmri), t(all.tau))
tau.temp.cor = cor(t(all.tau), t(all.temp))

#calculate coverage
cov.fmri = all.fmri != 0
cov.tau = all.tau != 0
cov.temp = all.temp != 0
cov.fmri.tau = cover(cov.fmri, cov.tau)
cov.tau.temp = cover(cov.tau, cov.temp)

cov.ft.ic = apply(cov.fmri.tau, 2, function(x) which(x==max(x)))
hist(diag(cov.fmri.tau[cov.ft.ic,]), xlab = 'Coverage',
     main = 'fMRIs vs. TAU', breaks = 6, cex.lab = 1.5, cex.main = 2, cex.axis = 1.3)
ind.ft = which(diag(cov.fmri.tau[cov.ft.ic,])>0.3)
cov.ft.ic[ind.ft]

cov.tt.ic = apply(cov.tau.temp, 2, function(x) which(x==max(x)))
hist(diag(cov.tau.temp[cov.tt.ic,]), xlab = 'Coverage', 
     main = 'TEMP vs. TAU', breaks = 10, cex.lab = 1.5, cex.main = 2, cex.axis = 1.3)
ind.tt = which(diag(cov.tau.temp[cov.tt.ic,])>0.3)
cov.tt.ic[ind.tt]

corrplot(cov.fmri.tau, mar=c(0,0,1,0), col.lim = c(0,1))
corrplot(cov.tau.temp, mar=c(0,0,1,0), col.lim = c(0,1))
cover = function(x, y, num = 25){
  covs = matrix(0, nrow = num, ncol = num)
  for(i in 1:num){
    for(j in 1:num){
      x1 = x[i,]
      y1 = y[j,]
      xy = x1 | y1
      xxyy = x1 * y1
      covs[i,j] = sum(xxyy)/sum(xy)
    }
  }
  return(covs)
}



fmri.temp.cor = cor(t(all.fmri), t(all.temp))
temp.mc.cor = cor(t(all.temp), t(mc))

corrplot(fmri.cor, title = 'fMRI', mar=c(0,0,1,0), col.lim = c(0,1))
corrplot(tau.cor,  title = 'TAU', mar=c(0,0,1,0))
corrplot(temp.cor, title = 'TEMP', mar=c(0,0,1,0))


corrplot(fmri.tau.cor, title = 'fMRI vs. TAU',mar=c(0,0,1,0))
corrplot(tau.temp.cor, title = 'TAU vs. TEMP', mar=c(0,0,1,0))

fmri.tau.cor.ic = apply(fmri.tau.cor, 2, function(x) which(abs(x)==max(abs(x))))
tau.temp.cor.ic = apply(tau.temp.cor, 2, function(x) which(abs(x)==max(abs(x))))
hist(abs(diag(fmri.tau.cor[fmri.tau.cor.ic,])), breaks = 6, xlab = 'Correlation', main = '')
hist(abs(diag(tau.temp.cor[tau.temp.cor.ic,])), breaks = 6, xlab = 'Correlation', main = '')

corrplot(fmri.temp.cor, title = 'fMRI vs. TEMP', mar=c(0,0,1,0))


corrplot(fmri.tau.cor)
corrplot(tau.temp.cor)
corrplot(mc.cor)
corrplot(adtau.cor)

#download aal2
# fn = "http://www.gin.cnrs.fr/wp-content/uploads/AAL3v2_for_SPM12.tar.gz"
# download.file(fn,destfile="tmp.tar.gz")
# untar("tmp.tar.gz",list=TRUE)
# untar("tmp.tar.gz")

aal = readNIfTI('aal2.nii')

dmn = array(0, dim=c(91, 109, 91, 4))

#PFC 21, 22
all.rs = c(21, 22, 39, 40, 71, 72, 69, 70)
t1 = aal
t1[which(t1 != 21 & t1 != 22)] = 0
dmn[,,,1] = t1


#PCC 39 40
t2 = aal
t2[which(t2 != 39 & t2 != 40)] = 0
dmn[,,,2] = t2

#precuneus 71, 72
t3 = aal
t3[which(t3 != 71 & t3 != 72)] = 0
dmn[,,,3] = t3

#angular gyrus 69, 70
t4 = aal
t4[which(t4 != 69 & t4 != 70)] = 0
dmn[,,,4] = t4

def = dmn

#DMN
d1 = aal
d1[which(!d1 %in% all.rs)]=0

#cover percentage of DMN
get_cov = function(img, num = 25){
  result = c()
  for(i in 1:num){
    cur = img[,,,i]
    o1 = sum((cur*t1)!=0)/sum(t1>0)
    o2 = sum((cur*t2)!=0)/sum(t2>0)
    o3 = sum((cur*t3)!=0)/sum(t3>0)
    o4 = sum((cur*t4)!=0)/sum(t4>0)
    result = rbind(result, c(o1, o2, o3, o4))
  }
  colnames(result) = c('PFC', 'PCC', 'PCUN', 'ANG')
  return(result)
}

vaica.dmn = get_cov(all.vaica)

fmri.dmn = get_cov(all.fmri)
tau.dmn = get_cov(all.tau)
temp.dmn = get_cov(all.temp)
mc.dmn = get_cov(mc)

dmn[dmn>0] = 30
dmn = as.nifti(dmn)
writeNIfTI(dmn, 'dmn', gzipped = T)


#overlap with each region
par(mfrow=c(3,1))
corrplot(t(fmri.dmn), title = 'fMRI', mar=c(0,0,1,0), col.lim = c(0,1))
corrplot(t(tau.dmn), title = 'TAU', mar=c(0,0,1,0), col.lim = c(0,1))
corrplot(t(temp.dmn), title = 'TEMP', mar=c(0,0,1,0), col.lim = c(0,1))
corrplot(t(vaica.dmn), title = 'VAICA', mar=c(0,0,1,0), col.lim = c(0,1))
corrplot(t(mc.dmn), title = 'GICs', mar=c(0,0,1,0), col.lim = c(0,1))


#find out IC associated most with the four ROIs
fmri.ic = apply(fmri.dmn, 2, function(x) which(x==max(x)))
tau.ic = apply(tau.dmn, 2, function(x) which(x==max(x)))
temp.ic = apply(temp.dmn, 2, function(x) which(x==max(x)))
mc.ic = apply(mc.dmn, 2, function(x) which(x==max(x)))
vaica.ic = apply(vaica.dmn, 2, function(x) which(x==max(x)))

va_nc = read.csv('va_nc.csv', header = F)
va_mci = read.csv('va_mci.csv', header = F)
va_ad = read.csv('va_ad.csv', header = F)
label =  c('NC', 'MCI', 'AD')
par(mfrow=c(5,5))
for(i in 1:25){
  boxplot(va_nc[,i]*10^7, va_mci[,i]*10^7, va_ad[,i]*10^7, names = label)
}

gift_fmri = readNIfTI('fmrimask_mean_component_ica_s1_.nii')
gift_fmri = trans_array(gift_fmri)
gift_fmri = trans_back(t(gift_fmri))
writeNIfTI(gift_fmri, 'gift_fmri', gzipped = F)

#use the mean of SUVR to shrink the size of TAU-PET
library(dplyr)
library(oro.nifti)
setwd('Y:/PhD/Fall 2020/RA')

cur_path = getwd()
#get the file names of TAU-PET
tau_path = paste0(cur_path, '/Data/MRI_TAU9')
file_names = dir(tau_path)
image_names = list()
for(i in file_names){
  cpath = paste0(tau_path, '/', i)
  sub_files = dir(cpath)
  sub_names = c()
  for(j in 1:length(sub_files)){
    spath = paste0(cpath, '/', sub_files[j])
    sub_names = c(sub_names, spath)
  }
  image_names[[i]] = sub_names
}


for(name in names(image_names)){
  tau.folder = paste0('Data/MRI_TAU9_3/', name)
  dir.create(tau.folder)
  
  files = image_names[[name]]
  num.files = length(files)
  
  for(i in 1:num.files){
    cur.img = readNIfTI(files[i])
    new.img = cur.img
    # for(d1 in 1:45){
    #   for(d2 in 1:54){
    #     for(d3 in 1:45){
    #       d1s = 2*d1-1
    #       d1e = 2*d1
    #       
    #       d2s = 2*d2-1
    #       d2e = 2*d2
    #       
    #       d3s = 2*d3-1
    #       d3e = 2*d3
    #       new.img[d1, d2, d3] = mean(cur.img[d1s:d1e, d2s:d2e, d3s:d3e])
    #     }
    #   }
    # }
    sub.file = paste0(tau.folder, '/sub', i)
    writeNIfTI(new.img, sub.file)
  }
  
}

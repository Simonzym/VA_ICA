library(fslr)
library(RNifti)
library(templateICAr)
library(stringr)
library(dplyr)
library(fastICA)
library(MASS)
library(pROC)
library(ggplot2)
library(corrplot)

tau_path = 'Y:/PhD/Thesis3/MRI_TAU12_MultipleVisits'
file_names = dir(tau_path)
image_names = list()
new_names = list()

for(i in file_names){
  cpath = paste0(tau_path, '/', i)
  sub_files = dir(cpath)
  sub_names = c()
  for(j in 1:length(sub_files)){
    spath = paste0(cpath, '/', sub_files[j])
    sub_names = c(sub_names, spath)
  }
  image_names[[i]] = sub_names
  if(substr(sub_files[1], 1, 1)=='I'){
    new_names[[i]] = sub_names
  }
}




setwd('Y:/')
for(i in names(new_names)){
  num = length(image_names[[i]])
  new_folder = paste0('TAU nii/', i, '/')
  cpath = paste0(tau_path, '/', i)
  all_files = list.files(cpath)
  from = paste0(cpath, '/')
  list.of.files <- list.files(from, full.names = T)
  for(j in 1:num){
    cur = readNIfTI(list.of.files[j])
    to = paste0('TAU nii/', i, '_', j)
    writeNIfTI(cur, to, gzipped = F)
  }
}

#longitudinal TAU
tmp = readNIfTI('fmri25Mask.nii')
tmps = readRDS('Y:/PhD/Thesis3/IC25/tmps.rds')
tau_mul = paste0('Y:/TAU nii/')
files = dir(tau_mul)
skull = readNIfTI('D:/Yimo/PhD/Thesis3/SkullStripped_MNITemplate_2mm.nii')
mask = skull>0
all_tau1 = c()
all_tau2 = c()
all_tau3 = c()
for(k in 1:200){
  print(k)
  file.name = paste0(tau_mul, files[k])
  cur = readNIfTI(file.name)
  cur = as.vector(cur)
  all_tau1 = rbind(all_tau1, cur[mask])
}

for(k in 201:400){
  print(k)
  file.name = paste0(tau_mul, files[k])
  cur = readNIfTI(file.name)
  cur = as.vector(cur)
  all_tau2 = rbind(all_tau2, cur[mask])
}

for(k in 401:length(files)){
  print(k)
  file.name = paste0(tau_mul, files[k])
  cur = readNIfTI(file.name)
  cur = as.vector(cur)
  all_tau3 = rbind(all_tau3, cur[mask])
}
all_tau = rbind(all_tau1, all_tau2, all_tau3)
dim(all_tau)
tau_result = templateICA(t(all_tau), Q2 = 0, tmps)
weights = tau_result$A
saveRDS(tau_result, 'Y:/PhD/Thesis3/IC25/multi_tau.rds')
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
tau_results = readRDS('Y:/PhD/Thesis3/IC25/multi_tau.rds')
weights = tau_result$A
tau_net = tau_result$subjICmean
tau25 = trans_back(tau_net)
#tau25 = trans_array(tau25)
tau25.dmn = get_cov(tau25)
corrplot(t(tau25.dmn), title = 'TAU', mar=c(0,0,1,0), col.lim = c(0,1))
#find out IC associated most with the four ROIs
tau25.ic = apply(tau25.dmn, 2, function(x) which(x==max(x)))

dt25 = data.frame(ic4 = weights[,4]*1000, ic11 = weights[,11]*1000, ic13 = weights[,13]*1000,
                  ic19 = weights[,19]*1000,
                  PTID = str_sub(files, 1, 10), sdate = scans$sdate)

multi.25 = dt25%>%inner_join(diagnosis, by = c('PTID'))%>%
  mutate(diag_diff = sdate-DIAGDATE)%>%filter(abs(diag_diff)<365/2)%>%
  filter(VISCODE2 != 'sc')%>%group_by(PTID)%>%filter(n()>1)%>%
  arrange(PTID, time)%>%
  mutate(time = ifelse(VISCODE2 == 'bl', 0, as.integer(str_sub(VISCODE2, 2, -1))))%>%
  mutate(lag_time = time - time[1])%>%ungroup(PTID)%>%
  mutate(DXCURREN = ifelse(DXCURREN==0, 'NC', ifelse(DXCURREN==1, 'MCI', 'AD')))

dd = multi.25%>%group_by(lag_time, DXCURREN)%>%
  summarise(ic13 = mean(ic13), ic11 = mean(ic11),ic4 = mean(ic4), ic19 = mean(ic19))
dd.p13 = ggplot(multi.25) + 
  geom_line(aes(lag_time, ic13, group = PTID), alpha = 0.1) +
  geom_line(data = dd, aes(x = lag_time, y = ic13, color =as.factor(DXCURREN)), alpha = .7, size = 2)+
  scale_color_manual(values=c('red', 'turquoise4', 'brown'))+ theme_bw() +
  labs(
    title = "IC 13",
    x = 'Time from First Scan (months)',
    y = "Weights",
    color = NULL
  )+
  theme(plot.title = element_text(hjust = 0.5),axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank())
dd.p11 = ggplot(multi.25) + 
  geom_line(aes(lag_time, ic11, group = PTID), alpha = 0.1) +
  geom_line(data = dd, aes(x = lag_time, y = ic11, color = as.factor(DXCURREN)), alpha = .7, size = 2)+
  scale_color_manual(values=c('red', 'turquoise4', 'brown'))+ theme_bw() +
  labs(
    title = "IC 11",
    x = 'Time from First Scan (months)',
    y = "Weights",
    color = NULL
  )+
  theme(plot.title = element_text(hjust = 0.5),axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank())
dd.p4 = ggplot(multi.25) + 
  geom_line(aes(lag_time, ic4, group = PTID), alpha = 0.1) +
  geom_line(data = dd, aes(x = lag_time, y = ic4, color = as.factor(DXCURREN)), alpha = .7, size = 2)+
  scale_color_manual(values=c('red', 'turquoise4', 'brown'))+ theme_bw() +
  labs(
    title = "IC 4",
    x = 'Time from First Scan (months)',
    y = "Weights",
    color = NULL
  )+
  theme(plot.title = element_text(hjust = 0.5), axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank())

dd.p19 = ggplot(multi.25) + 
  geom_line(aes(lag_time, ic19, group = PTID), alpha = 0.1) +
  geom_line(data = dd, aes(x = lag_time, y = ic19, color = as.factor(DXCURREN)), alpha = .7, size = 2)+
  scale_color_manual(values=c('red', 'turquoise4', 'brown'))+ theme_bw() +
  labs(
    title = "IC 19",
    x = 'Time from First Scan (months)',
    y = "Weights",
    color = NULL
  )+
  theme(plot.title = element_text(hjust = 0.5), axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank())
library(ggpubr)
library(patchwork)
ggarrange(dd.p4, dd.p11, dd.p13, dd.p19, common.legend = T, n.col = 1, n.row = 4, legend = 'right')
dd.p4+dd.p11+dd.p13+dd.p19+plot_layout(ncol = 1)&plot_annotation(title = 'TEMP')&theme(plot.title = element_text(hjust = 0.5))
#legend("bottomright", legend = c('IC4', 'IC8', 'IC16'), fill = c('red', 'blue', 'green'), cex = 0.5)
gift_25 = read.csv('gift25.csv', header = F)
gift_ic = read.csv('giftic.csv', header = F)
saveRDS(gift_ic, 'gift_ic.RDS')
gift_ic = as.matrix(gift_ic)
gift_back = trans_back(t(gift_ic))
gift25.dmn = get_cov(gift_back)
corrplot(t(gift25.dmn), title = 'GIFT25', mar=c(0,0,1,0), col.lim = c(0,1))
gift25.ic = apply(gift25.dmn, 2, function(x) which(x==max(x)))

dt.g25 = data.frame(ic20 = gift_25[,20], ic23 = gift_25[,23], ic13 = gift_25[,13],
                 ic16 = gift_25[,16] ,PTID = str_sub(files, 1, 10), sdate = scans$sdate)



par(mfrow=c(3,3), xpd=TRUE)
for(k in 1:num_sub){
  cur_dt = dt.g25%>%filter(PTID==unique(dt.g25$PTID)[k])
  max_ylim = max(c(cur_dt$ic7, cur_dt$ic12, cur_dt$ic16))
  min_ylim = min(c(cur_dt$ic7, cur_dt$ic12, cur_dt$ic16))
  dx = multi.info%>%filter(SubjectID == unique(dt.g25$PTID)[k])
  plot(as.integer(cur_dt$visit), cur_dt$ic7, type = 'l', col = 'red', 
       ylim = c(min_ylim, max_ylim), xlab = 'visit', ylab = 'weights', main = unique(dt.g25$PTID)[k])
  text(as.integer(cur_dt$visit),  cur_dt$ic7,  dx$DXCURREN,
       cex=0.65, pos=3,col="red") 
  lines(cur_dt$visit, cur_dt$ic12, col = 'blue')
  lines(cur_dt$visit, cur_dt$ic16, col = 'green')
}
par(mfrow=c(2,1))
corrplot(t(tau25.dmn),  mar=c(0,0,1,0), col.lim = c(0,1))
corrplot(t(gift25.dmn), mar=c(0,0,1,0), col.lim = c(0,1))

#viscode2
scans = data.frame()
for(name in names(new_names)){
  cur.file = new_names[[name]]
  cur.df = data.frame(PTID = name, ScanDate = str_sub(cur.file, -23, -16))
  scans = rbind(scans, cur.df)
}
scans = scans%>%mutate(sdate = as.Date(ScanDate, "%Y%m%d"))
#scan_date = read.csv('Y:/PhD/Thesis3/MRI_PET-matche_dates.csv')
#scan_date = scan_date%>%filter(SubjectID %in% names(new_names))%>%unique()


multi.info = dt.g25%>%inner_join(diagnosis, by = c('PTID'))%>%
  mutate(diag_diff = sdate-DIAGDATE)%>%filter(abs(diag_diff)<365/2)%>%
  filter(VISCODE2 != 'sc')%>%group_by(PTID)%>%filter(n()>1)%>%
  arrange(PTID, time)%>%
  mutate(time = ifelse(VISCODE2 == 'bl', 0, as.integer(str_sub(VISCODE2, 2, -1))))%>%
  mutate(lag_time = time - time[1])%>%ungroup(PTID)%>%
  mutate(DXCURREN = ifelse(DXCURREN==0, 'NC', ifelse(DXCURREN==1, 'MCI', 'AD')))


md = multi.info%>%group_by(lag_time, DXCURREN)%>%
  summarise(ic13 = mean(ic13), ic23 = mean(ic23),ic16 = mean(ic16), ic20 = mean(ic20))
p13 = ggplot(multi.info) + 
  geom_line(aes(lag_time, ic13, group = PTID), alpha = 0.1) +
  geom_line(data = md, aes(x = lag_time, y = ic13, color = as.factor(DXCURREN)), alpha = .7, size = 2)+
  scale_color_manual(values=c('red', 'turquoise4', 'brown'))+theme_bw() +
  labs(
    title = "IC 13",
    x = 'Time from First Scan (months)',
    y = "Weights",
    color = NULL
  )+
  theme(plot.title = element_text(hjust = 0.5),axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank())
p16 = ggplot(multi.info) + 
  geom_line(aes(lag_time, ic16, group = PTID), alpha = 0.1) +
  geom_line(data = md, aes(x = lag_time, y = ic16, color = as.factor(DXCURREN)), alpha = .7, size = 2)+
  scale_color_manual(values=c('red', 'turquoise4', 'brown'))+theme_bw() +
  labs(
    title = "IC 16",
    x = 'Time from First Scan (months)',
    y = "Weights",
    color = NULL
  )+
  theme(plot.title = element_text(hjust = 0.5),axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank())
p23 = ggplot(multi.info) + 
  geom_line(aes(lag_time, ic23, group = PTID), alpha = 0.1) +
  geom_line(data = md, aes(x = lag_time, y = ic23, color = as.factor(DXCURREN)), alpha = .7, size = 2)+
  scale_color_manual(values=c('red', 'turquoise4', 'brown'))+theme_bw() +
  labs(
    title = "IC 23",
    x = 'Time from First Scan (months)',
    y = "Weights",
    color = NULL
  )+
  theme(plot.title = element_text(hjust = 0.5),axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank())

p20 = ggplot(multi.info) + 
  geom_line(aes(lag_time, ic20, group = PTID), alpha = 0.1) +
  geom_line(data = md, aes(x = lag_time, y = ic20, color = as.factor(DXCURREN)), alpha = .7, size = 2)+
  scale_color_manual(values=c('red', 'turquoise4', 'brown'))+theme_bw() +
  labs(
    title = "IC 20",
    x = 'Time from First Scan (months)',
    y = "Weights",
    color = NULL
  )+
  theme(plot.title = element_text(hjust = 0.5),axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank())
ggarrange(p13, p16, p20, p23, common.legend = T)
p13+p16+p20+p23+plot_layout(ncol = 1)&plot_annotation(title = 'fastICA')&theme(plot.title = element_text(hjust = 0.5, size = 22))

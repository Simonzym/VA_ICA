#data generation
library(mvtnorm)
library(MASS)
library(sigmoid)
library(fastICA)
library(JADE)
#number of observations: n = 1200
#number of voxels: m = 1000
#number of independnet components: k = 3
setwd("Y:/PhD/Thesis2")
set.seed(321)

B = matrix(c(1,2,-1,sqrt(2)/2, 1/2,
             sqrt(3),-2,-1,1,sqrt(2),
             sqrt(5),1,-2,0,-1,
             0,1,-1,2,-sqrt(3),
             1,0,1,0,-1), nrow = 5, byrow = T)
write.csv(B, 'mix.csv', quote = F, row.names = F, col.names = F)

get_sum = function(num = 1){
  
  raw = read.csv(paste0('simHPC/test',num,'.csv'), header = F)
  colnames(raw) = NULL
  
  raw_s = read.csv(paste0('simHPC/sim',num, 'raw.csv'), header = F)
  s = read.csv(paste0('simHPC/sim', num, 'va.csv'), header = F)
  va_A = matrix(NA, nrow = 500, ncol = 5)
  for(i in 1:500){
    dt = data.frame(s)
    tmp = unlist(as.vector(raw[i,1:12846]))
    dt[,6] = tmp
    lr = lm(V6~.-1, data = dt)
    va_A[i, ] = coef(lr)
  }
  
  raw_A = matrix(NA, nrow = 500, ncol = 5)
  for(i in 1:500){
    dt = data.frame(raw_s)
    tmp = unlist(as.vector(raw[i,1:12846]))
    dt[,6] = tmp
    lr = lm(V6~.-1, data = dt)
    raw_A[i, ] = coef(lr)
  }
  
  e1va = c()
  e1raw = c()
  #va_A = t(read.csv('simHPC/sim4vamix.csv', header = F))
  #raw_A = t(read.csv('simHPC/sim4rawmix.csv', header = F))
  
  for(i in 1:100){
    start = 5*i-4
    end = 5*i 
    vaW = as.matrix(va_A[start:end, 1:5])
    rawW = as.matrix(raw_A[start:end, 1:5])
    e1va = c(e1va, amari.error(vaW, B))
    e1raw = c(e1raw, amari.error(rawW, B))
  }
  
  sim3 = data.frame(VA = e1va, fastICA = e1raw)
  return(sim3)
  
}
sim2 = get_sum(2)
sim3 = get_sum(3)
sim4 = get_sum(4)

apply(sim3, 2, mean)
apply(sim3, 2, sd)

apply(sim2, 2, mean)
apply(sim2, 2, sd)

apply(sim4, 2, mean)
apply(sim4, 2, sd)

t.test(sim2$VA, sim2$fastICA)
t.test(sim3$VA, sim3$fastICA)
t.test(sim4$VA, sim4$fastICA)
t.test(sim5$VA, sim5$fastICA)

par(mfrow=c(1,3))
boxplot(sim2, main = 'Scenario I')
boxplot(sim3, main = 'Scenario III')
boxplot(sim4, main = 'Scenario IV')

# template = readNIfTI('SkullStripped_MNITemplate_2mm.nii')
# mask = template>0
# path = 'D:/HCP_PTN1200_recon2/groupICA'
# file.name = 'groupICA_3T_HCP1200_MSMAll_d15.ica/melodic_IC_sum.nii'
# g_ics = readNIfTI(paste0(path, '/', file.name))
# GICs = c()
# setwd('Y:/PhD/Fall 2020/RA/')
# for(i in 1:15){
#   cur = as.vector(g_ics[,,,i])
#   GICs = rbind(GICs, cur[mask])
# }
# GICs = t(GICs)
# A = diag((1:15)/2, 15, 15)
# #use the HCP ICs
# #X = f(Y) + noise, y_i = a_i*S, x_i = f(y_i), S(k*m), Y(n*m), A(n*k)(unique), noise--standard normal
# 
# #first, fit the model of variational autoencoder to obtain the posterior distribution
# #secondly, draw Y from posterior distribution
# #then, using fastICA on y_i's trying to restore a_i
# 
# #things to change:
# #(1) begin with identity function of f(), then go with invertible function, and then noninvertible
# #(2) the number of independent components
# #(3) not neccessarily ICA assumption.
# 
# #use a_i for three types of observations (400*3)
# #train set.seed(123)
# #test set.seed(321)
# set.seed(123)
# 
# 
# k=15
# m=dim(GICs)[1]
# 
# 
# W.true=solve(A)
# 
# nsim=1
# Nsim=50
# maxit=100
# ### First let us fix the starting values of the mean sequences to 
# N0=19
# allX = c()
# S=GICs
# sg = sd(GICs)
# while(nsim<=Nsim){
#   print(nsim)
#   #noise from ICA assumption
#   noise1 = mvrnorm(m, mu = rep(0, 15), Sigma = diag(sg/10, 15, 15))
#   
#   
#   #add noise to the observation
#   noises = mvrnorm(m, mu = rep(0, 15), Sigma = diag(sg/20, 15, 15))
#   
#   ### Create X matrix
#   X=sigmoid(S%*%A+noise1)+noises
#   X=X-mean(X)
#   allX = rbind(allX, t(X))
#   nsim=nsim+1
# }
# setwd('Y:/PhD/Thesis2')
# dt = as.nifti(t(allX))
# write.csv(t(allX), 'simHPC/trainHPC3.csv')

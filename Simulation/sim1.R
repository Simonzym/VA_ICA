#data generation
library(mvtnorm)
library(JADE)
library(MASS)
library(fastICA)
#number of observations: n = 1200
#number of voxels: m = 1000
#number of independnet components: k = 3
setwd("Y:/PhD/Thesis2")


#X = f(Y) + noise, y_i = a_i*S, x_i = f(y_i), S(k*m), Y(n*m), A(n*k)(unique), noise--standard normal

#first, fit the model of variational autoencoder to obtain the posterior distribution
#secondly, draw Y from posterior distribution
#then, using fastICA on y_i's trying to restore a_i

#things to change:
#(1) begin with identity function of f(), then go with invertible function, and then noninvertible
#(2) the number of independent components
#(3) not neccessarily ICA assumption.

#use a_i for three types of observations (400*3)
#train set.seed(123)
#test set.seed(321)
k=3
m=1000
x.val=seq(-10,10,l=m)

##### The true standard deviations of the original sources
std=c(1,2,3)

A=matrix(c(1, 1, 1, 1, sqrt(3)/2, 0.5, 0.1, -0.5, sqrt(3)/2),3,3,byrow=TRUE)
W.true=solve(A)

nsim=1
Nsim=400
maxit=100
epsilon=10^(-5)
### First let us fix the starting values of the mean sequences to 
N0=19

get_data = function(seed = 123){
  set.seed(seed)
  allX = c()
while(nsim<=Nsim){
  
  ### Generate values from the densities of the hidden sources
  S=matrix(0,m,k)
  for(i in 1:m){

    u=runif(1,0,1)
    if(u<0.6){S[i,1]=rnorm(1,-4/5,1/5)}
    else     {S[i,1]=rnorm(1,6/5,1/5)}
    
    u=runif(1,0,1)
    if(u<0.5){S[i,2]=rnorm(1, -0.25, 1/2)}
    else{S[i,2]=rnorm(1, 0.25, sqrt(13/8))}
    
    u=runif(1,0,1)
    if(u<0.2)	 {S[i,3]=rnorm(1,5/sqrt(7.5),1/(2*sqrt(7.5)))}
    else {
      if(u<0.6){S[i,3]=rnorm(1,0,1/(2*sqrt(7.5)))}
      else 	 {S[i,3]=rnorm(1,-2.5/sqrt(7.5),1/(2*sqrt(7.5)))}}
  }
  
  #add noise to the observation
  noises = mvrnorm(1000, mu = c(0,0,0), Sigma = diag(0.1, 3,3))
  noise1 = mvrnorm(1000, mu = c(0,0,0), Sigma = diag(0.05, 3,3))
  
  ### Create X matrix
  X=S%*%A + noise1 
  allX = rbind(allX, t(X))
  nsim=nsim+1
}
  return(t(allX))
}
train = get_data(123)
test = get_data(321)
write.csv(train, 'SimData/train1.csv')
write.csv(test, 'SimData/test1.csv')

set.seed(321)

raw = read.csv('SimData/test1.csv', row.names = 1)


raw_s = read.csv('SimData/sim1raw.csv', header = F)
s = t(read.csv('SimData/sim1va.csv', header = F))

va_A = matrix(NA, nrow = 1200, ncol = 3)
for(i in 1:1200){
  dt = data.frame(s)
  dt[,4] = raw[,i]
  lr = lm(V4~.-1, data = dt)
  va_A[i, ] = coef(lr)
}

raw_A = matrix(NA, nrow = 1200, ncol = 3)
for(i in 1:1200){
  dt = data.frame(raw_s)
  dt[,4] = raw[,i]
  lr = lm(V4~.-1, data = dt)
  raw_A[i, ] = coef(lr)
}
A=matrix(c(1, 1, 1, 1, sqrt(3)/2, 0.5, 0.1, -0.5, sqrt(3)/2),3,3,byrow=TRUE)

e1va = c()
e1raw = c()
for(i in 1:400){
  start = 3*i-2
  end = 3*i 
  vaW = as.matrix(va_A[start:end, 1:3])
  rawW = as.matrix(raw_A[start:end, 1:3])
  e1va = c(e1va, amari.error(vaW, A))
  e1raw = c(e1raw, amari.error(rawW, A))
}
sim1 = data.frame(VA = e1va, fastICA = e1raw)
boxplot(sim1)

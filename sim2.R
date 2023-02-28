#data generation
library(mvtnorm)
library(MASS)
#number of observations: n = 1200
#number of voxels: m = 1000
#number of independnet components: k = 3
setwd("Y:/PhD/Thesis2/SimData")

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
set.seed(123)
k=3
m=1000
x.val=seq(-10,10,l=m)

##### The true stasndard deviations of the original sources
std=c(1,2,3)

A=matrix(c(1, 1, 1, 1, sqrt(3)/2, 0.5, 0.1, -0.5, sqrt(3)/2),3,3,byrow=TRUE)
W.true=solve(A)

nsim=1
Nsim=1200
maxit=100
epsilon=10^(-5)
### First let us fix the starting values of the mean sequences to 
N0=19
allX = c()

while(nsim<=Nsim){
  
  ### Generate values from the densities of the hidden sources
  S=matrix(0,m,k)
  for(i in 1:m){
    S[i,1]=rt(1,df=3)
    u=runif(1,0,1)
    if(u<0.5){S[i,2]=rnorm(1,-3,0.5)}
    else     {S[i,2]=rnorm(1,3,0.5)}
    u=runif(1,0,1)
    if(u<0.3)	 {S[i,3]=rnorm(1,-5,0.5)}
    else {
      if(u<0.7){S[i,3]=rnorm(1,0,0.5)}
      else 	 {S[i,3]=rnorm(1,5,0.5)}}
  }
  
  #add noise to the observation
  noises = rnorm(1000, mean = 0, sd = 0.1)
  
  ### Create X matrix
  inde = (nsim - 1)%/% 400 + 1
  X=A[inde, ]%*%t(S)+noises
  X=X-mean(X)
  allX = rbind(allX, X)
  nsim=nsim+1
}
write.csv(t(allX), 'train2.csv')



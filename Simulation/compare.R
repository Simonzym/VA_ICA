#calculate amari error for scenario 1
library(JADE)
setwd('Y:/PhD/Thesis2')


get_error = function(va.file, raw.file){
  
  sim1va = read.csv(va.file, header = F)
  sim1raw = read.csv(raw.file, header = F)
  #the true mixing matrix
  A=matrix(c(1,2,-1,sqrt(2)/2, 1/2,
             sqrt(3),-2,-1,1,sqrt(2),
             sqrt(5),1,-2,0,-1,
             0,1,-1,2,-sqrt(3),
             1,0,1,0,-1), nrow = 5, byrow = T)
  #calculate the amari error across subjects (400)
  e1va = c()
  e1raw = c()
  for(i in 1:100){
    start = 5*i-4
    end = 5*i 
    vaW = as.matrix(va_A[start:end, 1:3])
    rawW = as.matrix(raw_A[start:end, 1:3])
    e1va = c(e1va, amari.error(vaW, A))
    e1raw = c(e1raw, amari.error(rawW, A))
  }
  sim1 = data.frame(VA = e1va, fastICA = e1raw)
  return(sim1)
}
#sim1 = get_error('sim1va.csv', 'sim1raw.csv')
sim2 = get_error('simHPC/sim2vamix.csv', 'simHPC/sim2rawmix.csv')
sim3 = get_error('sim3va.csv', 'sim3raw.csv')
sim4 = get_error('sim4va.csv', 'sim4raw.csv')
apply(sim1, 2, mean)
apply(sim1, 2, sd)

apply(sim3, 2, mean)
apply(sim3, 2, sd)

apply(sim2, 2, mean)
apply(sim2, 2, sd)

apply(sim4, 2, mean)
apply(sim4, 2, sd)

boxplot(sim1)
boxplot(sim2)
boxplot(sim3)
boxplot(sim4)

par(mfrow=c(1,4))
boxplot(sim1, main = 'identity')
boxplot(sim31, main = 'sigmoid')
boxplot(sim32, main = 'sin')
boxplot(sim33, main = 'tanh')

par(mfrow = c(1, 4))
boxplot(sim1, main = 'Identity')
boxplot(sim3, main = 'Sigmoid')
boxplot(sim5, main = 'Sin')
boxplot(sim6, main = 'tanh')

par(mfrow = c(1, 5))
boxplot(sim3, main = 'd=0.1')
boxplot(sim31, main = 'd=2')
boxplot(sim32, main = 'd=5')
boxplot(sim33, main = 'd=10')
boxplot(sim34, main = 'd=20')

t.test(sim1$VA, sim1$fastICA)
t.test(sim31$VA, sim31$fastICA)
t.test(sim32$VA, sim32$fastICA)
t.test(sim33$VA, sim33$fastICA)


apply(sim1, 2, mean)
apply(sim1, 2, sd)

apply(sim31, 2, mean)
apply(sim31, 2, sd)

apply(sim32, 2, mean)
apply(sim32, 2, sd)

apply(sim33, 2, mean)
apply(sim33, 2, sd)

apply(sim5, 2, mean)
apply(sim5, 2, sd)

apply(sim6, 2, mean)
apply(sim6, 2, sd)

t.test(sim34$VA, sim34$fastICA)

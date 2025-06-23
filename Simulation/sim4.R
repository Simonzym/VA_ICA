#sim4
set.seed(321)
va = read.csv('mean31.csv', header = F)
result = fastICA(t(va), 3)

raw_s = read.csv('SimData/sim4raw.csv', header = F)
s = t(read.csv('SimData/sim4va.csv', header = F))

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
sim4 = data.frame(VA = e1va, fastICA = e1raw)
boxplot(sim4)


setwd("~/Desktop/Angel Martin")
datos <- read.table("pomeroy-2002-v2_database2.txt", header = TRUE, row.names = 1)

datos=as.data.frame(t(as.matrix(datos)))

dim(datos)


clase <- factor(c(rep("MD", 10), rep("Mglio", 10), rep("Rhab",10), rep("Ncer", 4), rep("PNET", 8)))

datos=cbind(clase, datos)


set.seed(100)
tr=round(nrow(datos)*0.7)
muestra=sample.int(nrow(datos), tr)
train=as.data.frame(datos[muestra,])
test=as.data.frame(datos[-muestra,])

dim(train)
dim(test)

library(FNN)
modelo=knn(train[,-1], test[,-1],k=10, train[,1], prob= T)
summary(modelo)

table(modelo, test[,1])




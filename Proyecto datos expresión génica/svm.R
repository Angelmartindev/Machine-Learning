BiocManager::install("multtest")
library (multtest)
data (golub)
datos <- as.data.frame(t(golub))
library ("e1071")

datos <- cbind (golub.cl, datos)
index <- 1:nrow(datos)

testindex <- sample(index, trunc(length(index)*30/100))

testset <- datos[testindex,]

trainset <- datos[-testindex,]

model  <- svm(golub.cl~., data = trainset, kernel = "polynomial",type = "C-classification") 
summary(model)

prediction <- predict(model, testset[,-1])
tab <- table(pred = prediction, true = testset[,1])
tab

citation ("e1071")

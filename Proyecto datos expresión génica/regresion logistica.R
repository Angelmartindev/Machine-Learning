BiocManager::install("multtest")
library (multtest)
data("golub")
View(golub)
x <- as.data.frame(t(golub))
View(y)
y <- golub.cl

#ACP y componentes principales almacenadas en 6 variables para poder incluirlas en el modelo
library(factoextra)
library(FactoMineR)

pca <- prcomp (x, center = T, scale = T)
fviz_screeplot(pca, addlabels = TRUE, ylim = c(0, 20))#elegimos 5 COMPONENTES


pca_bien <- get_pca_ind(pca, element=ind)

PCA1 <- pca_bien$coord [,1]
PCA2 <- pca_bien$coord [,2]
PCA3 <- pca_bien$coord [,3]
PCA4 <- pca_bien$coord [,4]
PCA5 <- pca_bien$coord [,5]

View(x)



#Si trabajamos con las componentes principales, los datos almacenados en las mismas son las cargas de cada una de las variables a las componentes, es decir, presenta 3051 valores
#por lo tanto no hay información de los individuos en las mismas. Si utilizo estas componentes como predictores en la regresión, no hay concordancia dimensional y no se puede
#reailzar el modelo

glm.fit <- glm(y ~ PCA1 + PCA2 + PCA3 + PCA4 + PCA5, data = x, family = binomial)

summary(glm.fit)

#Mediante biplot

library(MultBiplotR) 

bip=PCA.Biplot(x[,1:3051], dimension=5)
bip$RowCoordinates

bipfit=glm(y~bip$RowCoordinates,family = "binomial")

summary(bipfit)
anova(bipfit,test="Chisq")


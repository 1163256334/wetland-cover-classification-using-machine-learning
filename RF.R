library(e1071)
library(caret)
library(randomForest)

#load data
setwd("C:/Users/Administrator/Desktop")
modeldata=read.csv("modeldata.csv",stringsAsFactors = TRUE)
modeldata=modeldata[,c(-1)]
str(modeldata)
set.seed(1)
in_train=createDataPartition(modeldata$Class,p=0.7,list=FALSE)
trainset=modeldata[in_train,]
testset=modeldata[-in_train,]

#implementation of randomForest
?randomForest()
m.rf=randomForest(Class~., data=trainset,ntree=700,mtry=4,importance=TRUE)
m.rf$importance
varImpPlot(m.rf)
plot(m.rf$err.rate[,3])
plot(m.rf)

#performance
pre.rf=predict(m.rf,testset[,-17])
Matrix.rf=confusionMatrix(pre.rf,testset[,17])
Matrix.rf



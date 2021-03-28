library(e1071)
library(caret)
library(rJava)
library(RWeka)

#load data
setwd("C:/Users/Administrator/Desktop")
modeldata=read.csv("modeldata.csv",stringsAsFactors = TRUE)
modeldata=modeldata[,c(-1)]
str(modeldata)
set.seed(1)
in_train=createDataPartition(modeldata$Class,p=0.7,list=FALSE)
trainset=modeldata[in_train,]
testset=modeldata[-in_train,]

#implementation of ANN
MLP=make_Weka_classifier("weka/classifiers/functions/MultilayerPerceptron")
m.ann=MLP(Class~., data=trainset,control = Weka_control(L=0.4,M=0.7,D=TRUE,N=1000,H="a"))
m.ann

#performance
pre.ann=predict(m.ann,testset[,-17],type="class")
Matrix.ann=confusionMatrix(pre.ann,testset[,17])
Matrix.ann


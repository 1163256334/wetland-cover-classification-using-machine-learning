library(caret)

#load data
setwd("C:/Users/Administrator/Desktop")
modeldata=read.csv("modeldata.csv",stringsAsFactors = TRUE)

#decresing data size
for(i in 2:9){#i*0.1 represents the selection ratio 
  set.seed(i)
  ratio=createDataPartition(modeldata$Class,p=i*0.1,list=FALSE)
  assign(paste("modeldata",i,sep = ""),modeldata[ratio,])
}
str(modeldata9)
write.csv(modeldata2,file="rubost test-traning sample number/modeldata20%.csv")
write.csv(modeldata3,file="rubost test-traning sample number/modeldata30%.csv")
write.csv(modeldata4,file="rubost test-traning sample number/modeldata40%.csv")
write.csv(modeldata5,file="rubost test-traning sample number/modeldata50%.csv")
write.csv(modeldata6,file="rubost test-traning sample number/modeldata60%.csv")
write.csv(modeldata7,file="rubost test-traning sample number/modeldata70%.csv")
write.csv(modeldata8,file="rubost test-traning sample number/modeldata80%.csv")
write.csv(modeldata9,file="rubost test-traning sample number/modeldata90%.csv")


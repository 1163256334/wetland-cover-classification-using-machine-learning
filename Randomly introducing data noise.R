library(caret)

#load original data
setwd("C:/Users/Administrator/Desktop")
modeldata=read.csv("modeldata.csv",stringsAsFactors = TRUE)
str(modeldata)

#introducing noise####
#preparing training and testing data
for(i in 1:8){
  set.seed(i+100)
  in_train=createDataPartition(modeldata$Class,p=0.8,list=FALSE)
  assign(paste("training",i,sep = ""),modeldata[in_train,])
  assign(paste("testing",i,sep = ""),modeldata[-in_train,])
}
str(training1)
#selecting a cercain percentage data from training data to arise noise
for(i in 1:8){#i*0.1 represents the noise level 
  set.seed(i+100)
  in_train=createDataPartition(get(paste("training",i,sep = ""))$Class,p=i*0.1,list=FALSE)
  assign(paste("trainingN",i,sep = ""),get(paste("training",i,sep = ""))[in_train,])
  assign(paste("testingNoi",i,sep = ""),get(paste("training",i,sep = ""))[-in_train,])
}

#producing noise on trainingN

for(i in 1:8){
  set.seed(i+100)
  #length of trainingN
  NUM=nrow(get(paste("trainingN",i,sep = "")))
  #randrom class number that should be assigned
  rand=floor(runif(NUM,min=1,max=7))
  #classflame is used to store the noise class
  classflame=get(paste("trainingN",i,sep = ""))$Class
  for (j in 1:NUM){
    if (get(paste("trainingN",i,sep = ""))[j,]$Class=="AM")
      classflame[j]=switch(rand[j],"BR","CL","F","FP","MM","RL","SL")
    if (get(paste("trainingN",i,sep = ""))[j,]$Class=="BR")
      classflame[j]=switch(rand[j],"AM","CL","F","FP","MM","RL","SL")
    if (get(paste("trainingN",i,sep = ""))[j,]$Class=="CL")
      classflame[j]=switch(rand[j],"AM","BR","F","FP","MM","RL","SL")
    if (get(paste("trainingN",i,sep = ""))[j,]$Class=="F")
      classflame[j]=switch(rand[j],"AM","BR","CL","FP","MM","RL","SL")
    if (get(paste("trainingN",i,sep = ""))[j,]$Class=="FP")
      classflame[j]=switch(rand[j],"AM","BR","CL","F","MM","RL","SL")
    if (get(paste("trainingN",i,sep = ""))[j,]$Class=="MM")
      classflame[j]=switch(rand[j],"AM","BR","CL","F","FP","RL","SL")
    if (get(paste("trainingN",i,sep = ""))[j,]$Class=="RL")
      classflame[j]=switch(rand[j],"AM","BR","CL","F","FP","MM","SL")
    if (get(paste("trainingN",i,sep = ""))[j,]$Class=="SL")
      classflame[j]=switch(rand[j],"AM","BR","CL","F","FP","MM","RL")
  } 
  #assign noise to trainingNoii
  assign(paste("trainingNoi",i,sep = ""),data.frame(get(paste("trainingN",i,sep = ""))[,-18],Class=classflame))
}
str(trainingNoi1)
str(trainingNoise1)
#combining the noise data with the rest part of the training data to build new training data
for(i in 1:8){
  assign(paste("trainingNoise",i,sep = ""),rbind(get(paste("trainingNoi",i,sep = "")),get(paste("testingNoi",i,sep = ""))))
}
str(trainingNoise1)

#output
write.csv(trainingNoise1,file="rubost test-noise/trainingdata/trainingNoise10%.csv")
write.csv(trainingNoise2,file="rubost test-noise/trainingdata/trainingNoise20%.csv")
write.csv(trainingNoise3,file="rubost test-noise/trainingdata/trainingNoise30%.csv")
write.csv(trainingNoise4,file="rubost test-noise/trainingdata/trainingNoise40%.csv")
write.csv(trainingNoise5,file="rubost test-noise/trainingdata/trainingNoise50%.csv")
write.csv(trainingNoise6,file="rubost test-noise/trainingdata/trainingNoise60%.csv")
write.csv(trainingNoise7,file="rubost test-noise/trainingdata/trainingNoise70%.csv")
write.csv(trainingNoise8,file="rubost test-noise/trainingdata/trainingNoise80%.csv")


write.csv(testing1,file="rubost test-noise/testdata/testing1.csv")
write.csv(testing2,file="rubost test-noise/testdata/testing2.csv")
write.csv(testing3,file="rubost test-noise/testdata/testing3.csv")
write.csv(testing4,file="rubost test-noise/testdata/testing4.csv")
write.csv(testing5,file="rubost test-noise/testdata/testing5.csv")
write.csv(testing6,file="rubost test-noise/testdata/testing6.csv")
write.csv(testing7,file="rubost test-noise/testdata/testing7.csv")
write.csv(testing8,file="rubost test-noise/testdata/testing8.csv")



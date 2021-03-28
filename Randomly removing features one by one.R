NUM=11 #NUM is the number of features should be removed, which can be set by yourself
i=1
while (i<=10) { #10 is the replication times
  set.seed(i+NUM*NUM) # ensure removing is random
  assign(paste("ID",i,sep = ""),sample(1:16,NUM))
  i=i+1

}

#print the ID of features that should be removed
for (j in 1:10){
  print(sort(get(paste("ID",j,sep = ""))))
  
}

#load original data
setwd("C:/Users/Administrator/Desktop")
modeldata=read.csv("modeldata.csv",stringsAsFactors = TRUE)
modeldata=modeldata[,c(-1)]

#data after removing features
for (j in 1:10){
  assign(paste("modeldata",j,sep = ""),modeldata[,c(-get(paste("ID",j,sep = "")))])
}
str(modeldata1)#there are 10 dataset, namely modeldata1-10

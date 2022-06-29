options(scipen = 999)
df<-read.csv('c:/data/dacon/df_train.csv',encoding = "UTF-8")
result<-read.csv('C:/data/dacon/open/sample_submission.csv',encoding = "UTF-8")
test_area<-as.data.frame(read.csv('c:/data/dacon/test_area1.csv',encoding = "UTF-8"))
rownames(test_area)=NULL



# 회귀
test_area$이미지면적<-as.numeric(test_area$이미지면적)



model<-lm(무게~이미지면적,data=df)
summary(model)

k<-predict(model,newdata=test_area$이미지면적)
k<-abs(k)
k<-k+7.9025503211991595

result$leaf_weight<-k

write.csv(result,'C:/data/dacon/open/result_gap_r.csv', row.names = FALSE)

plot(무게~이미지면적,data=df)
abline(model,col='blue')

###################################
# 모든파일 가져와서 모델 돌려보기
df1<-read.csv('c:/data/dacon/result_area_gap.csv')
boxplot(df1[c(3:8,10:20)])

# 회귀나무
library(rpart)
library(rpart.plot)
df2<-df1[,-c(1,2,23)]

model_ct<-rpart(무게~.,data=df2,method='anova',cp=0,minsplit=nrow(df2)*0.05)
rpart.plot(model_ct)


# test data 
test_meta<-read.csv('c:/data/dacon/open/result_test_meta.csv')
test<-test_meta[,-1] # idx 제외외

k<-predict(model_ct,test)
k

result$leaf_weight<-k

write.csv(result,'C:/data/dacon/open/result_rpart_anova.csv', row.names = FALSE)
#######################################
#신경망  ㅅㅂ
#install.packages('neuralnet')
library(neuralnet)

for(i in 1:ncol(df2)){
  print(class(df2[,i]))
  }
df3<-na.omit(df2)

model_nn<-neuralnet(무게~.,data=df3,hidden=4,linear.output=T,stepmax = 1e7)
plot(model_nn)



k<-compute(model_nn,test)
summary(k)
k$net.result
########################################
# randomforest
library(randomForest)
model_rf<-randomForest(무게~.,data=df2,ntree=500,mtry=5,importance=T,na.action=na.omit)

k<-predict(model_rf,test)
k
result$leaf_weight<-k
write.csv(result,'C:/data/dacon/open/result_rf.csv', row.names = FALSE)
#########################
# xgboost
#install.packages("xgboost")
library(xgboost)

train_x<-data.matrix(df2[,-20])
train_y<-data.matrix(df2[,20])
xgb_train=xgb.DMatrix(data=train_x,label=train_y)
test_xgb<-data.matrix(test)



model_xgb<-xgboost(data=xgb_train,max.depth=12,nrounds = 7450,early_stopping_rounds = 10)
model_xgb
imp=xgb.importance(model=model_xgb)
xgb.plotimportance(imp)

k<-predict(model_xgb,test_xgb)
k
result$leaf_weight<-k
write.csv(result,'C:/data/dacon/open/result_xgb_of.csv',importance=T,row.names = FALSE)

#########################
# xgboost_scale
#install.packages("xgboost")
library(xgboost)

train_x<-data.matrix(scale(df2[,-20]))
train_y<-data.matrix(df2[,20])
xgb_train=xgb.DMatrix(data=train_x,label=train_y)
test_xgb<-data.matrix(scale(test))



model_xgb<-xgboost(data=xgb_train,max.depth=12,nrounds = 7450,early_stopping_rounds = 10)
imp=xgb.importance(model=model_xgb)
xgb.plot.importance(imp)

k<-predict(model_xgb,test_xgb)
result$leaf_weight<-k
write.csv(result,'C:/data/dacon/open/result_xgb_of.csv', row.names = FALSE)


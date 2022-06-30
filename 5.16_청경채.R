library(rpart)
library(rpart.plot)
library(neuralnet)
library(randomForest)
library(xgboost)

# 데이터 불러오기_yellow
df_train<-read.csv("C:/data/dacon/open/result_3m_meta_yellow.csv")
df_test<-read.csv("C:/data/dacon/open/test_3m_meta_yellow.csv")
result<-read.csv('C:/data/dacon/open/sample_submission.csv',encoding = "UTF-8") #제출용

# 필요없는 컬럼 제외
df_train<-df_train[-c(1,2,22)]
df_test<-df_test[-c(1,21)]

#########################
# xgboost with 3m
train_x<-data.matrix(df_train[,-20])
train_y<-data.matrix(df_train[,20])

xgb_train=xgb.DMatrix(data=train_x,label=train_y)

test_xgb<-data.matrix(df_test)

model_xgb<-xgboost(data=xgb_train,max.depth=15,eta=0.6,nrounds = 7450,early_stopping_rounds = 50,min_child_weight=3,max_delta_step=4)
imp=xgb.importance(model=model_xgb)

imp
xgb.plot.importance(imp)

k<-predict(model_xgb,test_xgb)
k
result$leaf_weight<-k
write.csv(result,'C:/data/dacon/open/result_xgb_3m_yellow.csv', row.names = FALSE)

#########################
# xgboost
train2<-df_train[,-c(22:57)]
test2<-df_test[,-c(21:56)]

train_x<-data.matrix(train2[,-21])
train_y<-data.matrix(train2[,21])
xgb_train=xgb.DMatrix(data=train_x,label=train_y)
test_xgb<-data.matrix(test2)

model_xgb<-xgboost(data=xgb_train,max.depth=12,nrounds = 50,early_stopping_rounds = 10)
model_xgb
imp=xgb.importance(model=model_xgb)

xgb.plot.importance(imp)

k<-predict(model_xgb,test_xgb)
k
result$leaf_weight<-k
write.csv(result,'C:/data/dacon/open/result_xgb_yellow.csv',row.names = FALSE)

############노란색 면적이 무게를 추가 시키는 듯
################변수를 초록면적과 노란면적 비율로 변환?
#########################
# xgboost
train2<-df_train[,-c(22:57)]
test2<-df_test[,-c(21:56)]

train2$yellow_ratio<-round(train2$image_area_yellow/train2$image_area_green,6)
test2$yellow_ratio<-round(test2$image_area_yellow/test2$image_area_green,6)

sum(is.na(test2$yellow_ratio))

train_x<-data.matrix(train2[,-21])
train_y<-data.matrix(train2[,21])

xgb_train=xgb.DMatrix(data=train_x,label=train_y)
test_xgb<-data.matrix(test2)

model_xgb<-xgboost(data=xgb_train,max.depth=5,nrounds = 105)
model_xgb
imp=xgb.importance(model=model_xgb)
xgb.plot.importance(imp)

k<-predict(model_xgb,test_xgb)
k
result$leaf_weight<-k
write.csv(result,'C:/data/dacon/open/result_xgb_yellow.csv',row.names = FALSE)


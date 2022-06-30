library(rpart)
library(rpart.plot)
library(neuralnet)
library(randomForest)
library(xgboost)

# 데이터 불러오기
df_train<-read.csv("C:/data/dacon/open/result_3m_meta.csv")
df_test<-read.csv("C:/data/dacon/open/test_3m_meta.csv")
result<-read.csv('C:/data/dacon/open/sample_submission.csv',encoding = "UTF-8") #제출용

colnames(df_train)
colnames(df_test)

# 필요없는 컬럼 제외
df_train<-df_train[-c(1,2)]
df_test<-df_test[-1]
#########################
# xgboost
train_x<-data.matrix(df_train[,-20])
train_y<-data.matrix(df_train[,20])

train_x<-data.matrix(df_train[,-20])
train_y<-data.matrix(df_train[,20])

xgb_train=xgb.DMatrix(data=train_x,label=train_y)

test_xgb<-data.matrix(df_test)



model_xgb<-xgboost(data=xgb_train,max.depth=15,eta=0.6,nrounds = 7450,early_stopping_rounds = 50,min_child_weight=3,max_delta_step=4)
k<-predict(model_xgb,test_xgb)
k
result$leaf_weight<-k
write.csv(result,'C:/data/dacon/open/result_xgb_3m.csv', row.names = FALSE)

####################################
#LGBM
install.packages("lightgbm")
library(lightgbm)

#
train.index<-sample(c(1:dim(df_train)[1]),dim(df_train)[1]*0.75)

train_x<-scale(df_train[train.index,-20])
train_y<-df_train[train.index,20]

test_x<-scale(df_train[-train.index,-20])
test_y<-df_train[-train.index,20]

dtrain = lgb.Dataset(train_x, label = train_y)
dtest = lgb.Dataset.create.valid(dtrain, test_x,label=test_y)

# define parameters
params = list(
  objective = "regression"
  , metric = "l2"
  , min_data = 1L
  , learning_rate = .3
)

# validataion data
valids = list(test = dtest)

# train model 
model = lgb.train(
  params = params
  , data = dtrain
  , nrounds = 5L
  , valids = valids
)


lgb.get.eval.result(model, "test", "l2")

pred_y = predict(model, test_x) 

# accuracy check
mse = mean((test_y - pred_y)^2)
mae = caret::MAE(test_y, pred_y)
rmse = caret::RMSE(test_y, pred_y)

cat("MSE: ", mse, "\nMAE: ", mae, "\nRMSE: ", rmse)

##################################################################
# xgboost_ scale
train_x<-data.matrix(scale(df_train[,-20]))
train_y<-data.matrix(df_train[,20])

train_x<-data.matrix(scale(df_train[,-20]))
train_y<-data.matrix(df_train[,20])

xgb_train=xgb.DMatrix(data=train_x,label=train_y)

test_xgb<-data.matrix(df_test)



model_xgb<-xgboost(data=xgb_train,max.depth=15,eta=0.6,nrounds = 7450,early_stopping_rounds = 50,min_child_weight=3,max_delta_step=4)
k<-predict(model_xgb,test_xgb)
k
result$leaf_weight<-k
write.csv(result,'C:/data/dacon/open/result_xgb_3m_scale.csv', row.names = FALSE)












## https://github.com/dmlc/xgboost/blob/master/R-package/demo/basic_walkthrough.R

library(data.table)
library(ROCR)
library(xgboost)
library(parallel)
library(Matrix)

set.seed(123)

d_train <- fread("train-1m.csv")
d_valid <- fread("valid.csv")
d_test <- fread("test.csv")


system.time({
  X_train_valid_test <- sparse.model.matrix(dep_delayed_15min ~ .-1, data = rbind(d_train, d_valid, d_test))
  n1 <- nrow(d_train)
  n2 <- nrow(d_valid)
  n3 <- nrow(d_test)
  X_train <- X_train_valid_test[1:n1,]
  X_valid <- X_train_valid_test[(n1+1):(n1+n2),]
  X_test <- X_train_valid_test[(n1+n2+1):(n1+n2+n3),]
})

dxgb_train <- xgb.DMatrix(data = X_train, label = ifelse(d_train$dep_delayed_15min=='Y',1,0))
dxgb_valid <- xgb.DMatrix(data = X_valid, label = ifelse(d_valid$dep_delayed_15min=='Y',1,0))
dxgb_test  <- xgb.DMatrix(data = X_test,  label = ifelse(d_test$dep_delayed_15min =='Y',1,0))



system.time({
  n_proc <- detectCores()
  md <- xgb.train(data = dxgb_train, nthread = n_proc, 
          objective = "binary:logistic", nrounds = 100, 
          max_depth = 20, eta = 0.1, 
          min_child_weight = 1, subsample = 0.5,
          watchlist = list(valid = dxgb_valid, train = dxgb_train), 
          eval_metric = "auc",
          early.stop.round = 10, print.every.n = 10)
})


phat <- predict(md, newdata = X_test)
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
print(performance(rocr_pred, "auc"))

summary(phat)

head(xgb.dump(md),20)
tail(xgb.dump(md),20)

## xgb.importance(model = md)  ## hangs



system.time({
  n_proc <- detectCores()
  md <- xgb.cv(data = dxgb_train, nthread = n_proc, 
            objective = "binary:logistic", nrounds = 100, 
            nfold = 5,
            max_depth = 20, eta = 0.1, 
            min_child_weight = 1, subsample = 0.5,
            eval_metric = "auc",
            early.stop.round = 10, print.every.n = 10)
})

tail(md)


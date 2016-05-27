
library(xgboost)
library(Matrix)

set.seed(123)

d_train <- readr::read_csv("train-0.1m.csv")
d_test <- readr::read_csv("test.csv")

X_train <- sparse.model.matrix(dep_delayed_15min ~ .-1, data = d_train)

system.time({
  n_proc <- parallel::detectCores()
  md <- xgboost(data = X_train, label = ifelse(d_train$dep_delayed_15min=='Y',1,0),
                 nthread = n_proc, nround = 1, max_depth = 20,
                 num_parallel_tree = 100, subsample = 0.632,
                 colsample_bytree = 1/sqrt(length(X_train@x)/nrow(X_train)))
})



X_test <- X_train[1,]
system.time({
  print(predict(md, newdata = X_test))
})
##Error in xgb.DMatrix(newdata) : 
##  xgb.DMatrix: does not support to construct from  double


X_test <- X_train[1:2,]
system.time({
  print(predict(md, newdata = X_test)[1])
})
## 1-30ms   0.4219182


X_test <- Matrix(X_train[1,], nrow=1, dimnames = list(1,names(X_train[1,])))
system.time({
  print(predict(md, newdata = X_test))
})
## 0.4219182


d_test[1,1:(ncol(d_test)-1)]

system.time({
  X_test <- Matrix::sparse.model.matrix(dep_delayed_15min ~ .-1, 
              data = rbind(d_test[1,],d_train))[1:2,]
})
## 400ms
system.time({
  print(predict(md, newdata = X_test)[1])
})
## 0.3925396


col_cl <- sapply(d_test[1,1:(ncol(d_test)-1)],class)
col_str <- names(col_cl[col_cl=="character"])
col_nostr <- names(col_cl[col_cl!="character"])

system.time({
  names1 <- paste0(col_str, d_test[1,col_str])
  X_str <- Matrix(rep(1,length(names1)), nrow=1, dimnames = list(1,names1), sparse = TRUE)
  X_test <- cbind(X_str, as.matrix(d_test[1,col_nostr]))
})
## 1-5ms
system.time({
  print(predict(md, newdata = X_test))
})
## 1-30ms  0.4380869



## https://github.com/dmlc/xgboost/issues/318






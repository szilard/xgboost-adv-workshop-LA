
library(h2o)

h2o.init(max_mem_size="60g", nthreads=-1)


dx_train <- h2o.importFile("train-1m.csv")
dx_test <- h2o.importFile("test.csv")


Xnames <- names(dx_train)[which(names(dx_train)!="dep_delayed_15min")]

system.time({
  md <- h2o.randomForest(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, ntrees = 100)
})


system.time({
  print(h2o.performance(md, dx_test)@metrics$AUC)
})


phat <- as.data.frame(h2o.predict(md, dx_test))[["Y"]]

summary(phat)
##    Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
##0.004756 0.266600 0.362400 0.378300 0.474500 0.965600 

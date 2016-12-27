# Function for Cross Validation using lxgboost

xgboost_cv <- function(Training,Target,cv=5,metric="logloss",importance=0)
{
  Training$order <- seq(1, nrow(Training))
  Training$result <- as.factor(Target)
  class(Training$exposure)
  set.seed(123)
  Training$randomCV <- floor(runif(nrow(Training), 1, (cv+1)))
  param <- list(  objective           = "binary:logistic", 
                  booster             = "gbtree",
                  eval_metric         = "auc",  # maximizing for auc
                  eta                 = 0.002,   # learning rate - Number of Trees
                  max_depth           = 7,      # maximum depth of a tree
                  subsample           = .9,     # subsample ratio of the training instance
                  colsample_bytree    = .87,    # subsample ratio of columns 
                  min_child_weight    = 1,      # minimum sum of instance weight (defualt)
                  scale_pos_weight    = 1       # helps convergance bc dataset is unbalanced
  ) 
  
  # cross validation
  cat(cv, "-fold Cross Validation\n", sep = "")
  Training$result = as.numeric(Training$result)
  Training$result = Training$result -1
  for (i in 1:cv)
  {
    Train_Fold <- subset(Training, randomCV != i, select = -c(order, randomCV))
    Validation <- subset(Training, randomCV == i) 
    # building model
    Train_Fold$result = as.numeric(Train_Fold$result)
    train_new <- sparse.model.matrix(Train_Fold$result ~ ., data = Train_Fold)
    dtrain <- xgb.DMatrix(data=train_new, label=Train_Fold$result)
    model_xgb <- xgb.train(   params              = param, 
                              data                = dtrain, 
                              nrounds             = 50, 
                              verbose             = 1,
                              maximize            = FALSE
    )
    Validation$target <- -1
    # predicting on validation data
    testing <- sparse.model.matrix(target ~ ., data = Validation)
    preds <- predict(model_xgb, testing)
    Validation$pred_xgb <- preds
    Validation$target <- NULL
    # Printing CV scores
    cat("CV Fold-", i, " ", metric, ": ", score(Validation$result, Validation$pred_xgb, metric), "\n", sep = "")
  } 
}
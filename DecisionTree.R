# Function for Cross Validation using lDecision Trees

DecisionTree_cv <- function(Training,Target,cv=5,metric="logloss",importance=0)
{
  Training$order <- seq(1, nrow(Training))
  Training$result <- as.numeric(Target)
  Training$randomCV <- floor(runif(nrow(Training), 1, (cv+1)))
  # cross validation
  cat(cv, "-fold Cross Validation\n", sep = "")
  for (i in 1:cv)
  {
    Train_Fold <- subset(Training, randomCV != i, select = -c(order, randomCV))
    Validation <- subset(Training, randomCV == i) 
    # building model
    model_Dtree = tree(result ~., data=Train_Fold)
    # predicting on validation data
    pred_dt <- predict(model_Dtree, Validation)
    Validation <- cbind(Validation, pred_dt)
    # Printing CV scores
    cat("CV Fold-", i, " ", metric, ": ", score(Validation$result, Validation$pred_dt, metric), "\n", sep = "")
  } 
}
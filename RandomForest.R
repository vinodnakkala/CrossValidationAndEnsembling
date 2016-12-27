# Function for Cross Validation using Random Forests

RandomForestRegression_CV <- function(Training,Target,cv=5,ntree=50,nodesize=5,seed=123,metric="logloss")
{
  Training$order <- seq(1, nrow(Training))
  Training$result <- as.factor(Target)
  set.seed(seed)
  Training$randomCV <- floor(runif(nrow(Training), 1, (cv+1)))
  cat(cv, "-fold Cross Validation\n", sep = "")
  for (i in 1:cv)
  {
    Train_Fold <- subset(Training, randomCV != i, select = -c(order, randomCV))
    Validation <- subset(Training, randomCV == i) 
    model_rf <- randomForest(result ~., data = Train_Fold, ntree = ntree, nodesize = nodesize, type="prob")
    prediction <- predict(model_rf, Validation,type="prob")
    prediction = as.data.frame(prediction)
    colnames(prediction) <- c("prob0", "prob1")
    pred_rf = prediction$prob1
    Validation <- cbind(Validation, pred_rf)
    # Printing CV scores
    cat("CV Fold-", i, " ", metric, ": ", score(Validation$result, Validation$pred_rf, metric), "\n", sep = "")
  } 
}
# Function for Cross Validation using logistic regression

LogisticRegression_CV <- function(Training,Target,cv=5,metric="logloss",importance=0)
{
  cat("Preparing Data for LogisticRegression\n")
  Training$order <- seq(1, nrow(Training))
  Training$result <- as.factor(Target)
  Training$randomCV <- floor(runif(nrow(Training), 1, (cv+1)))
  # cross validation
  cat(cv, "-fold Cross Validation\n", sep = "")
  for (i in 1:cv)
  {
    Train_Fold <- subset(Training, randomCV != i, select = -c(order, randomCV))
    Validation <- subset(Training, randomCV == i) 
    # building LogisticRegression model
    model_lr <- glm(result ~., data=Train_Fold, family=binomial())
    # predicting on validation data
    pred_lr <- predict(model_lr, Validation, type="response")
    Validation <- cbind(Validation, pred_lr)
    # Printing CV scores
    cat("CV Fold-", i, " ", metric, ": ", score(Validation$result, Validation$pred_lr, metric), "\n", sep = "")
  } 
}



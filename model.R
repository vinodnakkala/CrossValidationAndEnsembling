options(warn=-1)
suppressWarnings(library(pROC))
library(caTools)
library(sampling)
library(randomForest)
library(caret)
library(pROC)
library(tree)
library(Matrix)
library(xgboost)
library(data.table)
library(xgboost)

#### Change the woking directory to point it to the folder where r files and dataset is stored ######
#setwd("F:/R_Class/Project2/Project2/")
setwd("C:/Users/nakka/Downloads/R/Proj2")
total = read.csv("dataset.csv")
##### set the seed to make your partition reproductible
total$outcome = as.factor(total$outcome)
dim(total)
summary(total$outcome)
set.seed(123)
total_sampled <- sample(seq_len(nrow(total)), size = 300)
test <- total[total_sampled, ]
train_valid <- total[-total_sampled, ]
dim(train_valid)
dim(test)
sample_train <- sample(seq_len(nrow(train_valid)), size = 200)
train <- train_valid[sample_train, ]
valid <- train_valid[-sample_train, ]

########################### Sampled for training, validation and testing ####################################
train$outcome = as.factor(train$outcome)
class(train$outcome)
model_lr <- glm(outcome ~., data=train, family=binomial())
pred_lr <- predict(model_lr, valid)
valid$prob_lr = pred_lr
valid$predict_lr = ifelse(valid$prob_lr>0.5,1,0)
accuracy <- confusionMatrix(valid$predict_lr,valid$outcome)
accuracy_logisticRegression <- accuracy$overall['Accuracy']


############################################### Random Forests ############################################
set.seed(123)
model_rf <- randomForest(outcome ~., data = train, ntree = 25, nodesize = 4, type="prob")
pred_rf <- predict(model_rf, valid)
valid$predict_rf = pred_rf
accuracy <- confusionMatrix(valid$predict_rf,valid$outcome)
accuracy_randomForest <- accuracy$overall['Accuracy']


################################################# Decision Trees ###########################################
fit = tree(outcome~., data=train)
Pred_dt <- predict(fit, valid)
Pred_dt = as.data.frame(Pred_dt)
colnames(Pred_dt) <- c("prob0","prob1")
valid$prob_dt = Pred_dt$prob1
valid$predict_dt = ifelse(valid$prob_dt>0.5,1,0)
accuracy <- confusionMatrix(valid$predict_dt,valid$outcome)
accuracy_decisionTree <- accuracy$overall['Accuracy']

################################################## xgboost #################################################
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


train$outcome = as.numeric(as.character(train$outcome))
#train$outcome = train$outcome-1
set.seed(123)
train_new <- sparse.model.matrix(train$outcome ~ ., data = train)
dtrain <- xgb.DMatrix(data=train_new, label=train$outcome)
model_xgb <- xgb.train(   params              = param, 
                          data                = dtrain, 
                          nrounds             = 25, 
                          verbose             = 1,
                          maximize            = FALSE
)
valid$target <- -1

######## predicting on validation data ####
testing <- sparse.model.matrix(target ~ ., data = valid)
preds_xgb <- predict(model_xgb, testing)
valid$prob_xgb <- preds_xgb
valid$target <- NULL
valid$predict_xgb = ifelse(valid$prob_xgb>0.5,1,0)
accuracy <- confusionMatrix(valid$predict_xgb,valid$outcome)
accuracy_xgb <- accuracy$overall['Accuracy']


############################ Confuson matrix for base models ############################################
# Seeing the accuracies in Validation data
accuracy_logisticRegression
accuracy_decisionTree
accuracy_randomForest
accuracy_xgb

#################################### Chose xgboost and predict on test data ###########################
test$target <- -1
testing <- sparse.model.matrix(target ~ ., data = test)
preds_xgb <- predict(model_xgb, testing)
test$prob_xgb <- preds_xgb
test$target <- NULL
test$predict_xgb = ifelse(test$prob_xgb>0.5,1,0)
accuracy <- confusionMatrix(test$predict_xgb,test$outcome)
accuracy_final_xgb <- accuracy$overall['Accuracy']
accuracy_final_xgb
################################ We obtain accuracy of 79.33% on test data ################################

dim(train)
dim(test)
dim(train_valid)
## Writing a function to generate the metrics if actual outcome and probabilities are passed as arguments
score <- function(a,b,metric)
{
  switch(metric,
       accuracy = sum(abs(a-b)<=0.5)/length(a),
       auc = auc(a,b),
       logloss = -(sum(log(1-b[a==0])) + sum(log(b[a==1])))/length(a),
       mae = sum(abs(a-b))/length(a),
       precision = length(a[a==b])/length(a),
       rmse = sqrt(sum((a-b)^2)/length(a)),
       rmspe = sqrt(sum(((a-b)/a)^2)/length(a)))           
}
################################ Fit the models using Cross Validation  #################################
#############################################################################
########################### Data Preparation for Logistic Regression  #######################

test$outcome = as.numeric(as.character(test$outcome))
trainedTarget = train_valid$outcome
ActualTargets <- test$outcome
test$outcome <- NULL
train_valid$outcome <- NULL
############################################# Logistic Regression #########################################
source("./LogisticRegression.R")
model_lr <- LogisticRegression_CV(train_valid,trainedTarget, cv=5, metric="logloss")
dim(train_valid)
dim(test)
summary(trainedTarget)
################################################ Decision Tree ##########################################
dim(test)
dim(train_valid)
#trainedTarget = train_valid$outcome
#train_valid$outcome <- NULL
source("./DecisionTree.R")
model_dt <- DecisionTree_cv(train_valid,trainedTarget,cv=5,metric="accuracy")

############################################ Random Forest #############################################
##################################### Data Preparation for Random Forest ######################
dim(total)
set.seed(123)
#total_sampled <- sample(seq_len(nrow(total)), size = 200)
#test <- total[total_sampled, ]
#train_valid <- total[-total_sampled, ]
dim(train_valid)
dim(test)

############################## Random Forest Modelling with CV ##########################################
source("./RandomForest.R")
model_rf_1 <- RandomForestRegression_CV(train_valid,trainedTarget,cv=5,ntree=25,nodesize=5,seed=234,metric="logloss")

############################################### XG Boost ##################################################
source("./xgBoost.R")
model_xgb <- xgboost_cv(train_valid,trainedTarget,cv=5,metric="logloss")
########################################################################
##### Building Random Forest for entire dataset as Cross vaidation scores of Random forest looks best
train_valid$outcome = trainedTarget
class(train_valid$outcome)
train_valid$outcome = as.factor(train_valid$outcome)
model_rf <- randomForest(outcome ~., data = train_valid, ntree = 25, nodesize = 5, type="prob")
pred_rf <- predict(model_rf, test)
test$outcomeProbs = pred_rf
#View(test)
test$predicted = pred_rf
confusionMatrix(test$predicted,ActualTargets)
rf_cv = test$predicted
test$predicted =NULL
test$outcomeProbs = NULL
# Consistent CV scores and giving 86% accuracy


########################## Building other models for Ensemble modelling #################

### Logistic Regression
model_lr <- glm(train_valid$outcome ~., data=train_valid, family=binomial())
pred_lr <- predict(model_lr, test, type="response")
test$outcomeProbs = pred_lr
ActualTargets = ActualTargets
test$predicted = ifelse(test$outcomeProbs>0.5,1,0)
accuracy <- confusionMatrix(test$predicted,ActualTargets)
accuracy_lr <- accuracy$overall['Accuracy']
lr_cv = test$predicted

## Building xgboost for entire dataset
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


train_valid$result = as.numeric(trainedTarget)
summary(train_valid$result)
train_valid$result = train_valid$result-1
train_new <- sparse.model.matrix(train_valid$result ~ ., data = train_valid)
dtrain <- xgb.DMatrix(data=train_new, label=train_valid$result)
model_xgb <- xgb.train(   params              = param, 
                          data                = dtrain, 
                          nrounds             = 50, 
                          verbose             = 1,
                          maximize            = FALSE
)
test$target <- -1
##### predicting on test data
testing <- sparse.model.matrix(target ~ ., data = test)
preds <- predict(model_xgb, testing)
test$pred_xgb <- preds
test$target <- NULL
test$predicted = ifelse(test$pred_xgb>0.5,1,0)
xgb_cv = test$predicted
test$predicted =NULL
test$outcomeProbs = NULL
train_valid$result = NULL
## Building Decision Tree for entire dataset
train_valid$outcome = trainedTarget
#class(train_valid$outcome)
train_valid$outcome=as.factor(train_valid$outcome)
fit = tree(outcome~., data=train_valid)
Pred_dt <- predict(fit, test)
Pred_dt = as.data.frame(Pred_dt)
colnames(Pred_dt) <- c("prob0","prob1")
test$outcomeProbs = Pred_dt$prob1
test$predicted = ifelse(test$outcomeProbs>0.5,1,0)
dt_cv = test$predicted
test$predicted =NULL
test$outcomeProbs = NULL
dim(test)
dim(train_valid)
train_valid$outcome <- NULL
########################################### Ensemble modelling using voting ################################
temp = cbind(ActualTargets,lr_cv,dt_cv,rf_cv,xgb_cv)
temp=as.data.frame(temp)
colnames(temp) <- c("outcome","lr_cv","dt_cv","rf_cv","xgb_cv")
temp$rf_cv = as.numeric(rf_cv) 
temp$rf_cv = temp$rf_cv-1
temp$Ensemble = ifelse(temp$dt_cv+temp$rf_cv+temp$xgb_cv >= 2,1,0)
confusionMatrix(temp$Ensemble,ActualTargets)
#Voting Ensemble is giving 88.33% Accuracy 

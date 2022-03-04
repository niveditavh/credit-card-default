set.seed(123)
ccdata <- read.csv("E:/DMML_Datasets/credit_card/UCI_Credit_Card.csv", header = TRUE)
summary(ccdata)
str(ccdata)

########################target variable check ###################

#checking target variable 0->no_default 1->default
#The percentage of "Default" class is 22%, so the data imbalance is not significant.
counts <- table(ccdata$default.payment.next.month)
barplot(counts, main="Target Variable visualization",ylab="Count",
        xlab="Default payment",col=c("darkblue","lightsalmon2"))






library(ggplot2) 
#0->no_default 1->default
credit_education_default<-table(credit_card_data$default.payment.next.month,credit_card_data$EDUCATION)
barplot(credit_education_default, xlab='Education',ylab='Count',main="Education with credit_default_count",
        col=c("darkblue","brown")
        ,legend=rownames(credit_education_default), args.legend = list(x = "topleft"),beside=T)

#1->male 2->female
#female has more default
credit_gender_default<-table(credit_card_data$default.payment.next.month,credit_card_data$SEX)
barplot(credit_gender_default, xlab='Gender',ylab='Count',main="Gender with credit_default_count",
        col=c("darkblue","brown")
        ,legend=rownames(credit_education_default), args.legend = list(x = "topleft"),beside=T)




# Age distribution by default status
#As age increases to 30, the probability of default increases. Meanwhile, when clients are over 30, the probability decreases when aging.
credit_age_default <- ggplot(credit_card_data, aes(credit_card_data$AGE))+ 
  geom_density(aes(fill=factor(credit_card_data$default.payment.next.month)), alpha=0.3) + 
  labs(title="Distribution of Age by Default Payment Status", 
       x="Age",fill="Default Payment Next Month") + 
  scale_y_continuous(expand = c(0,0))+
  scale_fill_manual(values=c("blue","#ADB6B5"))
plot(credit_age_default)



# Credit limit by default status
#Clients with lower amounts tend to default. Especially those with credit amount around 50000 default most.
credit_limit_default <- ggplot(credit_card_data, aes(credit_card_data$LIMIT_BAL))+ 
  geom_density(aes(fill=factor(credit_card_data$default.payment.next.month)), alpha=0.3) + 
  labs(title="Distribution of Credit Limit by Default Payment Status", 
       x="Credit Limit",fill="Default Payment Next Month") + 
  scale_y_continuous(expand = c(0,0))+
  scale_fill_manual(values=c("blue","#ADB6B5"))
plot(credit_limit_default)




#######checking for multicollinearity########
library(corrplot)

corrplot(cor(ccdata[1:21]),method = "number")








##########removing columns #############
ccdata<-ccdata[,c(-1,-3,-5)]
str(ccdata)
######checking null######
sum(is.na(ccdata))
########check the outliers#######
boxplot(ccdata)
dim(ccdata)
######removing extreme outliers #############
ccdata<-ccdata[ccdata$LIMIT_BAL<1000000,]
ccdata<-ccdata[!ccdata$BILL_AMT1<0,]
ccdata<-ccdata[!ccdata$BILL_AMT1>7e+05,]
ccdata<-ccdata[!ccdata$BILL_AMT2>6e+05,]
ccdata<-ccdata[!ccdata$BILL_AMT3>6e+05,]
ccdata<-ccdata[!ccdata$BILL_AMT3<0,]
ccdata<-ccdata[!ccdata$BILL_AMT4>6e+05,]
ccdata<-ccdata[!ccdata$BILL_AMT5>6e+05,]
ccdata<-ccdata[!ccdata$BILL_AMT5<0,]
ccdata<-ccdata[!ccdata$BILL_AMT6>6e+05,]
ccdata<-ccdata[!ccdata$BILL_AMT6<0,]
ccdata<-ccdata[!ccdata$PAY_AMT1>3e+05,]
ccdata<-ccdata[!ccdata$PAY_AMT2>5e+05,]
ccdata<-ccdata[!ccdata$PAY_AMT3>4e+05,]
ccdata<-ccdata[!ccdata$PAY_AMT4>3e+05,]
ccdata<-ccdata[!ccdata$PAY_AMT5>350000,]
ccdata<-ccdata[!ccdata$PAY_AMT6>3e+05,]
boxplot(ccdata)


dim(ccdata)




#######data partition #########


library(caret)
set.seed(2341)
trainIndex <- createDataPartition(ccdata$default.payment.next.month, p = 0.80, list = FALSE)

trainData.dt <- ccdata[trainIndex, ]

testData.dt <- ccdata[-trainIndex, ]

dim(trainData.dt)

dim(testData.dt)






#######svm #################
############### it will take 20mins #########################

library(e1071)
model_svm<-svm(`default.payment.next.month` ~ ., data=trainData.dt,cost=100,gamma=1)
summary(model_svm)


######### predect test svm ###########
pred_test_svm<-predict(model_svm,testData.dt)
library(caret)
confusionMatrix(as.factor(testData.dt$`default.payment.next.month`),as.factor(pred_test_svm))
#accuracy - 75%
#sensitivity - 81%
#specificity -44%
#kappa -21%
#balanced accuracy - 62%
#auc - 59.6%


auc_svm <- roc(as.numeric(testData.dt$`default.payment.next.month`),as.numeric(pred_test_svm), plot = TRUE, legacy.axes= TRUE, 
               percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
               col="#377eb8", lwd=4, print.auc = TRUE)




##############decision tree#########
#install.packages('C50')
library(C50)
trainData.dt$`default.payment.next.month`<-as.factor(trainData.dt$`default.payment.next.month`)
class(trainData.dt$`default.payment.next.month`)

model_c50<-C5.0(`default.payment.next.month` ~ ., data=trainData.dt)
summary(model_c50)

######### predicting test decision tree #######
pred_test_d<-predict(model_c50,testData.dt)
library(caret)
confusionMatrix(as.factor(testData.dt$`default.payment.next.month`),as.factor(pred_test_d))
#accuracy - 82%
#sensitivity - 83%
#specificity - 68%
#kappa - 37%
#balanced accuracy - 76%
#auc -65.7%

library(pROC)
par(pty ="s")
auc_decision <- roc(as.numeric(testData.dt$`default.payment.next.month`),as.numeric(pred_test_d), plot = TRUE, legacy.axes= TRUE, 
                    percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
                    col="#377eb8", lwd=4, print.auc = TRUE)


#############xtreme gradient boosting ##########
#install.packages('xgboost')
library(xgboost)
x<-trainData.dt[,1:21]
x
y<-trainData.dt[,22]
y
model_xgb<-xgboost(data=as.matrix(x),label = as.matrix(y),nrounds = 100)


######### predicting on test xtreme gradient boosting ############

x_pred<-testData.dt[,1:21]

y_pred<-testData.dt[,22]


pred_xtreme<-predict(model_xgb,as.matrix(x_pred))
pred_test_ext<-round(pred_xtreme)
library(caret)
confusionMatrix(as.factor(testData.dt$`default.payment.next.month`),as.factor(pred_test_ext))

#accuracy - 81%
#sensitivity -84%
#specificity - 64%
#kappa - 37%
#balanced accuracy - 66%


auc_xtreme_boosting <- roc(as.numeric(testData.dt$`default.payment.next.month`),as.numeric(pred_test_ext), plot = TRUE, legacy.axes= TRUE, 
                           percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
                           col="#377eb8", lwd=4, print.auc = TRUE)


################################## shuffle the data ################################################################################
set.seed(42)


trainIndex <- createDataPartition(ccdata$default.payment.next.month, p = 0.80, list = FALSE)

train2 <- ccdata[trainIndex, ]

test2 <- ccdata[-trainIndex, ]

dim(train2)

dim(test2)





######### predect test svm ###########
pred_test_svm1<-predict(model_svm1,test2)
library(caret)
confusionMatrix(as.factor(test2$`default.payment.next.month`),as.factor(pred_test_svm1))
#accuracy - 75%
#sensitivity - 81%
#specificity -44%
#kappa -21%
#balanced accuracy - 62%
#auc - 59.6%


auc_svm <- roc(as.numeric(test2$`default.payment.next.month`),as.numeric(pred_test_svm1), plot = TRUE, legacy.axes= TRUE, 
               percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
               col="#377eb8", lwd=4, print.auc = TRUE)



############## shuffle decision tree #########
#install.packages('C50')
library(C50)
train2$`default.payment.next.month`<-as.factor(train2$`default.payment.next.month`)
class(train2$`default.payment.next.month`)

model_c501<-C5.0(`default.payment.next.month` ~ ., data=train2)
summary(model_c501)


######### predicting test decision tree #######
pred_test_d1<-predict(model_c501,test2)
library(caret)
confusionMatrix(as.factor(test2$`default.payment.next.month`),as.factor(pred_test_d1))
#accuracy - 82%
#sensitivity - 83%
#specificity - 68%
#kappa - 37%
#balanced accuracy - 76%
#auc -65.7%

library(pROC)
par(pty ="s")
auc_decision <- roc(as.numeric(test2$`default.payment.next.month`),as.numeric(pred_test_d1), plot = TRUE, legacy.axes= TRUE, 
                    percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
                    col="#377eb8", lwd=4, print.auc = TRUE)

######### predicting on test xtreme gradient boosting ############

x_pred<-test2[,1:21]

y_pred<-test2[,22]


pred_xtreme1<-predict(model_xgb,as.matrix(x_pred))
pred_test_ext1<-round(pred_xtreme1)
library(caret)
confusionMatrix(as.factor(test2$`default.payment.next.month`),as.factor(pred_test_ext1))

#accuracy - 81%
#sensitivity -84%
#specificity - 64%
#kappa - 37%
#balanced accuracy - 66%


auc_xtreme_boosting <- roc(as.numeric(test2$`default.payment.next.month`),as.numeric(pred_test_ext1), plot = TRUE, legacy.axes= TRUE, 
                           percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
                           col="#377eb8", lwd=4, print.auc = TRUE)





########################## conclusion #################################################

#accuracy,sensitivity,specificity for first iteration
#SVM- 75% 81% 68%
#Decision Tree - 82% 83% 68%
#Xtreme Gradient Boosting - 81% 84% 64%

#accuracy,sensitivity,specificity for second iteration
#SVM- 76% 81% 44%
#Decision Tree -82% 84% 68%
#Xtreme Gradient Boosting -81% 84% 66%

########### future #############

#dimentionality techniques can be applied because it has more feature variables

































#########binary logistic regression ########

model_bin<-glm(default.payment.next.month ~ ., data=trainData.dt,family = binomial(link="logit"))

#using exp()function we are converting it into odds ratio for interpreting results,whenn all coefficients are zero
#the odds ratio of payment default is 0.29,when limt_bal increases by a unit the payment defaulters increases by 0.99,
#similary other variables are also interpreted in odds ratio

exp(model_bin$coefficients)


############ predicting on test data ###########

pred_model_bin<-ifelse(model_bin$fitted.values>0.5,1,0)
library(caret)
confusionMatrix(as.factor(testData.dt$`default.payment.next.month`),as.factor(pred_model_bin))
# accuracy - 81%
# sensitivity(true positive) - 82%
# specificity(true neagtive) - 72%
# balanced accuracy - 76%
## auc 0.64


library(pROC)
par(pty ="s")
auc_lm <- roc(as.numeric(testData.dt$`default.payment.next.month`),as.numeric(pred_model_bin), plot = TRUE, legacy.axes= TRUE, 
              percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
              col="#377eb8", lwd=4, print.auc = TRUE)


library(ROCR)

pr<-prediction(as.numeric(pred_model_bin),as.numeric(testData.dt$`default.payment.next.month`))
auc<-performance(pr,measure = "auc")
auc<-auc@y.values
auc



prf<-performance(pr,measure = "tpr",x.measure = "fpr")
plot(prf)





#########naive bayes method #########

trainData.dt$default.payment.next.month<-as.factor(trainData.dt$default.payment.next.month)
class(trainData.dt$default.payment.next.month)

library(naiveBayes)
library(e1071)
library(caTools)
library(caret)
model_nb<-naiveBayes(`default.payment.next.month` ~ ., data=trainData.dt)


########## predicting on test #############
pred_test_n<-predict(model_nb,testData.dt)
library(caret)
confusionMatrix(as.factor(testData.dt$`default.payment.next.month`),as.factor(pred_test_n))

# accuracy - 76%
# sensitivity(true positive) - 87%
# specificity(true neagtive) - 47%
# balanced accuracy - 67%

#areaundercurve - 70.4%



library(pROC)
par(pty ="s")
auc_lm <- roc(as.numeric(testData.dt$`default.payment.next.month`),as.numeric(pred_test_n), plot = TRUE, legacy.axes= TRUE, 
              percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
              col="#377eb8", lwd=4, print.auc = TRUE)


############### CART(CLASSIFICATION AND REGRESSION TREES) Decision treee ##################
library(rpart)
model_cart<-rpart(`default.payment.next.month` ~ ., data=trainData.dt)
summary(model_cart)


####### prediction on test data ########
pred_test_cart<-predict(model_cart,testData.dt,type="class")
library(caret)
confusionMatrix(as.factor(testData.dt$`default.payment.next.month`),as.factor(pred_test_cart))

# accuracy - 82%
# sensitivity(true positive) - 83%
# specificity(true neagtive) - 70%
#kappa -37%
# balanced accuracy - 77%

#areaundercurve - 65%

library(pROC)
par(pty ="s")
auc_cart <- roc(as.numeric(testData.dt$`default.payment.next.month`),as.numeric(pred_test_cart), plot = TRUE, legacy.axes= TRUE, 
              percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
              col="#377eb8", lwd=4, print.auc = TRUE)






###########cart bagging ensemble model#############
library(ipred)
model_cartbag<-bagging(`default.payment.next.month` ~ ., data=trainData.dt)
varImp(model_cartbag)

######### cart bagging predicting on test ############

pred_test_cart_bagging<-predict(model_cartbag,testData.dt,type="class")
library(caret)
#predicted<-ifelse(pred_train_cart> 0.5,1,0)
confusionMatrix(as.factor(testData.dt$`default.payment.next.month`),as.factor(pred_test_cart_bagging))

#accuracy - 81%
#sensitivity - 83%
#specificity - 62%
#balanced accuracy - 73%
#auc - 65.4%

auc_cart_bagging <- roc(as.numeric(testData.dt$`default.payment.next.month`),as.numeric(pred_test_cart_bagging), plot = TRUE, legacy.axes= TRUE, 
               percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
               col="#377eb8", lwd=4, print.auc = TRUE)




############## adaptive boosting ensemble model ############
#install.packages('ada')
library(ada)
model_ada<-ada(`default.payment.next.month` ~ ., data=trainData.dt,loss="exponential",type="discrete",iter=100)
summary(model_ada)
plot(model_ada)


##### predict test adaptive boosting ######
pred_test_ada<-predict(model_ada,testData.dt)
library(caret)
confusionMatrix(as.factor(testData.dt$`default.payment.next.month`),as.factor(pred_test_ada))

#Here in boosting it classifies weak learners and makes them strong learners using decision stump.
#It performs 100 iterations by exponential method .
#It has a train and test error of 18% 

#accuracy - 82%
#sensitivity - 84%
#specificity - 68%
#balanced accuracy - 76%
#auc - 66.3%


auc_ada_boosting <- roc(as.numeric(testData.dt$`default.payment.next.month`),as.numeric(pred_test_ada), plot = TRUE, legacy.axes= TRUE, 
                        percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
                        col="#377eb8", lwd=4, print.auc = TRUE)







######## shuffled naive bayes ##############
train2$default.payment.next.month<-as.factor(train2$default.payment.next.month)
class(train2$default.payment.next.month)


library(e1071)
library(caTools)
library(caret)
model_nb<-naiveBayes(`default.payment.next.month` ~ ., data=train2)


########## predicting on test #############
pred_test_n1<-predict(model_nb,test2)
library(caret)
confusionMatrix(as.factor(test2$`default.payment.next.month`),as.factor(pred_test_n1))

# accuracy - 76%
# sensitivity(true positive) - 87%
# specificity(true neagtive) - 47%
# balanced accuracy - 67%

#areaundercurve - 70.4%



library(pROC)
par(pty ="s")
auc_lm <- roc(as.numeric(test2$`default.payment.next.month`),as.numeric(pred_test_n1), plot = TRUE, legacy.axes= TRUE, 
              percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
              col="#377eb8", lwd=4, print.auc = TRUE)






############ shuffled extreme boost ###########

library(xgboost)
x<-train2[,1:21]

y<-train2[,22]

model_xgb<-xgboost(data=as.matrix(x),label = as.matrix(y),nrounds = 100)





#############  shuffled ada boosting #############


library(ada)
model_ada1<-ada(`default.payment.next.month` ~ ., data=train2,loss="exponential",type="discrete",iter=100)
summary(model_ada1)
plot(model_ada1)


##### predict test adaptive boosting ######
pred_test_ada1<-predict(model_ada1,test2)
library(caret)
confusionMatrix(as.factor(test2$`default.payment.next.month`),as.factor(pred_test_ada1))

#Here in boosting it classifies weak learners and makes them strong learners using decision stump.
#It performs 100 iterations by exponential method .
#It has a train and test error of 18% 

#accuracy - 82%
#sensitivity - 84%
#specificity - 68%
#balanced accuracy - 76%
#auc - 66.3%


auc_ada_boosting <- roc(as.numeric(test2$`default.payment.next.month`),as.numeric(pred_test_ada1), plot = TRUE, legacy.axes= TRUE, 
                        percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
                        col="#377eb8", lwd=4, print.auc = TRUE)



###########shuffle cart bagging ensemble model#############
library(ipred)
model_cartbag1<-bagging(`default.payment.next.month` ~ ., data=train2)
varImp(model_cartbag1)

######### cart bagging predicting on test ############

pred_test_cart_bagging1<-predict(model_cartbag1,test2,type="class")
library(caret)
#predicted<-ifelse(pred_train_cart> 0.5,1,0)
confusionMatrix(as.factor(test2$`default.payment.next.month`),as.factor(pred_test_cart_bagging1))

#accuracy - 81%
#sensitivity - 83%
#specificity - 62%
#balanced accuracy - 73%
#auc - 65.4%

auc_cart_bagging <- roc(as.numeric(test2$`default.payment.next.month`),as.numeric(pred_test_cart_bagging1), plot = TRUE, legacy.axes= TRUE, 
                        percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
                        col="#377eb8", lwd=4, print.auc = TRUE)


####### shuffle   svm #################

library(e1071)
model_svm1<-svm(`default.payment.next.month` ~ ., data=train2,cost=100,gamma=1)
summary(model_svm1)


######### predect test svm ###########
pred_test_svm1<-predict(model_svm1,test2)
library(caret)
confusionMatrix(as.factor(test2$`default.payment.next.month`),as.factor(pred_test_svm1))
#accuracy - 75%
#sensitivity - 81%
#specificity -44%
#kappa -21%
#balanced accuracy - 62%
#auc - 59.6%


auc_svm <- roc(as.numeric(test2$`default.payment.next.month`),as.numeric(pred_test_svm1), plot = TRUE, legacy.axes= TRUE, 
               percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
               col="#377eb8", lwd=4, print.auc = TRUE)



############## shuffle decision tree #########
#install.packages('C50')
library(C50)
train2$`default.payment.next.month`<-as.factor(train2$`default.payment.next.month`)
class(train2$`default.payment.next.month`)

model_c501<-C5.0(`default.payment.next.month` ~ ., data=train2)
summary(model_c501)


######### predicting test decision tree #######
pred_test_d1<-predict(model_c501,test2)
library(caret)
confusionMatrix(as.factor(test2$`default.payment.next.month`),as.factor(pred_test_d1))
#accuracy - 82%
#sensitivity - 83%
#specificity - 68%
#kappa - 37%
#balanced accuracy - 76%
#auc -65.7%

library(pROC)
par(pty ="s")
auc_decision <- roc(as.numeric(test2$`default.payment.next.month`),as.numeric(pred_test_d1), plot = TRUE, legacy.axes= TRUE, 
                    percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
                    col="#377eb8", lwd=4, print.auc = TRUE)


############### CART(CLASSIFICATION AND REGRESSION TREES) Decision treee ##################
library(rpart)
model_cart1<-rpart(`default.payment.next.month` ~ ., data=train2,v = 20)
summary(model_cart1)

####### prediction on test data ########
pred_test_cart1<-predict(model_cart1,test2,type="class")
library(caret)
confusionMatrix(as.factor(test2$`default.payment.next.month`),as.factor(pred_test_cart1))

# accuracy - 82%
# sensitivity(true positive) - 83%
# specificity(true neagtive) - 70%
#kappa -37%
# balanced accuracy - 77%

#areaundercurve - 65%

library(pROC)
par(pty ="s")
auc_cart <- roc(as.numeric(test2$`default.payment.next.month`),as.numeric(pred_test_cart1), plot = TRUE, legacy.axes= TRUE, 
                percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
                col="#377eb8", lwd=4, print.auc = TRUE)



############# knn ###########

library(caret)
library(e1071)

trctrl <- trainControl(method = "repeatedcv",
                       number = 10,
                       repeats = 3)
set.seed(3333)
#install.packages('e1071', dependencies=TRUE)

trainData.dt$default.payment.next.month<-as.factor(trainData.dt$default.payment.next.month)
class(trainData.dt$default.payment.next.month)

# Run kNN Classifier in package caret
knn_fit  <- train(`default.payment.next.month` ~ .,
                  data = trainData.dt,
                  method = "knn",
                  trControl = trctrl,
                  preProcess = c("center", "scale"),
                  tuneLength = 10)
# kNN model summary
knn_fit 




# predicting the test set observations
#kNNPred <- predict(knn_fit, testSet, type = "raw")
kNNPred <- predict(knn_fit, testData.dt, type = "prob")

predictedLabels <- ifelse(kNNPred$Yes > 0.2,"Yes","No")
# ordering the levels
predictedLabels <- ordered(predictedLabels, levels = c("Yes", "No"))

confusionMatrix(as.factor(testData.dt$`default.payment.next.month`),as.factor(predictedLabels))

# confusion matrix
cm = table(Predicted = predictedLabels, Actual = testData.dt$default.payment.next.month)
print(cm)

#Calculate stats from confusion matrix
tp = cm[1,1]
fp = cm[1,2]
fn = cm[2,1]
tn = cm[2,2]

Accuracy = 100*(tp + tn)/(tp + fp + fn + tn)
Sensitivity = 100*(tp)/(tp + fn)
Specificity = 100*(tn)/(fp + tn)
Precision = 100*(tp)/(tp + fp)







kNNPred <- predict(knn_fit, testData.dt,type = "raw")
kNNPredObj <- prediction(as.numeric(kNNPred),as.numeric(testData.dt$`default.payment.next.month`))
kNNPe <- performance(kNNPredObj, "tpr","fpr")
kNNPe
d<-ave(as.factor(testData.dt$`default.payment.next.month`),as.factor(kNNPredObj),FUN=mean)
table(as.factor(testData.dt$`default.payment.next.month`),as.factor(kNNPredObj))
sensitivity(data01, ref01)

library(caret)
confusionMatrix(as.factor(ave(testData.dt$`default.payment.next.month`),as.factor(kNNPredObj),FUN = mean))





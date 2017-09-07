######################################################

# STEP 1: START #

install.packages("gmodels")
install.packages("Hmisc")
install.packages("pROC")
install.packages("ResourceSelection")
install.packages("car")
install.packages("caret")
install.packages("dplyr")
library(gmodels)
library(VIF)
library(caret)
library(Hmisc)
library(pROC)
library(ResourceSelection)
library(car)
library(caret)
library(dplyr)
install.packages("InformationValue")
library(InformationValue)

cat("\014") # Clearing the screen

# STEP 1: END #

# STEP 2: START #

# Setting the working directory - 
getwd()
setwd("C:/YYYYYY/AMMA 2017/Data/bank") #This working directory is the folder where all the bank data is stored

# reading client datasets
df.client <- read.csv('bank_client.csv')
str(df.client)

# reading other attributes
df.attr <- read.csv('bank_other_attributes.csv')
str(df.attr)

# reading campaign data
df.campaign <- read.csv('latest_campaign.csv')
str(df.campaign)

# reading campaign outcome
df.campOutcome <- read.csv('campaign_outcome.csv')
str(df.campOutcome)

# Create campaign data by joining all tables together
df.temp1 <- merge(df.client, df.campaign, by = 'Cust_id', all.x = TRUE)
df.temp2 <- merge(df.temp1, df.attr, by = 'Cust_id', all.x = TRUE)
df.data <- merge(df.temp2, df.campOutcome, by = 'Cust_id', all.x = TRUE)
length(unique(df.data$Cust_id)) == nrow(df.data) #checking for any duplicate customer ID

# clearing out temporary tables
rm(df.temp1,df.temp2)

# see few observations of merged dataset
head(df.data)

# STEP 2: END #

# STEP 3: START #

###### Crude Model Code Start ######

# see a quick summary view of the dataset
summary(df.data)

# see the tables structure
str(df.data)

# check the response rate
CrossTable(df.data$y)

# split the data into training and test
set.seed(1234) # for reproducibility
df.data$rand <- runif(nrow(df.data))
df.train <- df.data[df.data$rand <= 0.7,]
df.test <- df.data[df.data$rand > 0.7,]
nrow(df.train)
nrow(df.test)

# how the categorical variables are distributed and are related with target outcome
CrossTable(df.train$job, df.train$y)
CrossTable(df.train$marital, df.train$y)
CrossTable(df.train$education, df.train$y)
CrossTable(df.train$default, df.train$y)
CrossTable(df.train$housing, df.train$y)
CrossTable(df.train$loan, df.train$y)
CrossTable(df.train$poutcome, df.train$y)

# let see how the numerical variables are distributed
hist(df.train$age)
hist(df.train$balance)
hist(df.train$duration)
hist(df.train$campaign)
hist(df.train$pdays)
hist(df.train$previous)
describe(df.train[c("age", "balance", "duration", "campaign", "pdays", "previous")])

# running a full model  #

df.train$yact = ifelse(df.train$y == 'yes',1,0)
full.model <- glm(formula = yact ~ age + balance + duration + campaign + pdays + previous +
                    job + marital + education + default + housing + loan + poutcome, 
                  data=df.train, family = binomial)
summary(full.model)

# check for vif
fit <- lm(formula <- yact ~ age + balance + duration + campaign + pdays + previous +
            job + marital + education + default + housing + loan + poutcome, 
          data=df.train)
vif(fit)

# automated variable selection - Backward
backward <- step(full.model, direction = 'backward')
summary(backward)

# training probabilities and roc
df.train$prob = predict(full.model, type=c("response"))
class(df.train)
nrow(df.train)
q <- roc(y ~ prob, data = df.train)
plot(q)
auc(q)

# variable importance
varImp(full.model, scale = FALSE)

# confusion matrix
df.train$ypred = ifelse(df.train$prob>=.5,'pred_yes','pred_no')
table(df.train$ypred,df.train$y)

#ks plot
ks_plot(actuals=df.train$y, predictedScores=df.train$ypred)

###### Crude Model Code End ######

# STEP 3: END #

########## RY - code Start ##############

# STEP 4: START #

View(df.data)

# Loading df.data into df.data_final
df.data_final <- df.data
df.data_final$yact = ifelse(df.data$y == 'yes',1,0) #Loading 1s for 'yes' and 0s for 'no'
nrow(df.data_final)

#Removing every row with Not-Available entries
df.data_final <- df.data_final[!apply(df.data_final[,c("age", "balance", "duration", "campaign", "pdays", "previous", "job","marital", "education", "default", "housing", "loan", "poutcome")], 1, anyNA),]
nrow(df.data_final)
View(df.data_final)

set.seed(1234) # for reproducibility
df.data_final$rand <- runif(nrow(df.data_final))

#Training set = 85% of the entire data set #Test set = 15% of the entire data set
df.train_rymodel <- df.data_final[df.data_final$rand <= 0.85,]
df.test_rymodel <- df.data_final[df.data_final$rand > 0.85,]
nrow(df.train_rymodel)

#garbage collection to remove garbage from memory - to ensure memory overload doesn't happen
gc()

#Building a tentative model - with all the insignificant variables
result_tentative_trainrymodel <- glm(formula = yact ~ age + balance + duration + campaign + pdays + previous +
                                      job + marital + education + default + housing + loan + poutcome, 
                                    data=df.train_rymodel, family = binomial)
summary(result_tentative_trainrymodel)

# The process of removing insignificant variables one at a time based on their p-values
# removing the insignificant variable job unknown as it has the highest insignificance value

df.train_rymodel_onlysig <- df.train_rymodel[df.train_rymodel$job!="unknown",]
result_tentative_trainrymodel_sig1 <- glm(formula = yact ~ age + balance + duration + campaign + previous + job + marital + education + housing + loan + poutcome, data=df.train_rymodel_onlysig, family = binomial)

df.test_rymodel_onlysig <- df.test_rymodel[df.test_rymodel$job!="unknown",]

summary(result_tentative_trainrymodel_sig1)


# removing the insignificant variable pdays as it has the highest insignificance value

df.train_rymodel_onlysig$pdays <-NULL
result_tentative_trainrymodel_sig1 <- glm(formula = yact ~ age + balance + duration + campaign + previous + job + marital + education + housing + loan + poutcome, data=df.train_rymodel_onlysig, family = binomial)

df.test_rymodel_onlysig$pdays <-NULL

summary(result_tentative_trainrymodel_sig1)

# removing the insignificant variable jobmanagement as it has the highest insignificance value
df.train_rymodel_onlysig <- df.train_rymodel_onlysig[df.train_rymodel_onlysig$job!="management",]
result_tentative_trainrymodel_sig1 <- glm(formula = yact ~ age + balance + duration + campaign + previous + job + marital + education + housing + loan + poutcome, data=df.train_rymodel_onlysig, family = binomial)

df.test_rymodel_onlysig <- df.test_rymodel_onlysig[df.test_rymodel_onlysig$job!="management",]

summary(result_tentative_trainrymodel_sig1)

# removing the insignificant variable jobentrepreneur as it has the highest insignificance value

df.train_rymodel_onlysig <- df.train_rymodel_onlysig[df.train_rymodel_onlysig$job!="entrepreneur",]
result_tentative_trainrymodel_sig1 <- glm(formula = yact ~ age + balance + duration + campaign + previous + job + marital + education + housing + loan + poutcome, data=df.train_rymodel_onlysig, family = binomial)

df.test_rymodel_onlysig <- df.test_rymodel_onlysig[df.test_rymodel_onlysig$job!="entrepreneur",]

summary(result_tentative_trainrymodel_sig1)

# removing the insignificant variable maritalsingle as it has the highest insignificance value

df.train_rymodel_onlysig <- df.train_rymodel_onlysig[df.train_rymodel_onlysig$marital!="single",]
result_tentative_trainrymodel_sig1 <- glm(formula = yact ~ age + balance + duration + campaign + previous + job + marital + education + housing + loan + poutcome, data=df.train_rymodel_onlysig, family = binomial)

df.test_rymodel_onlysig <- df.test_rymodel_onlysig[df.test_rymodel_onlysig$marital!="single",]

summary(result_tentative_trainrymodel_sig1)

# removing the insignificant variable defaultyes as it has the highest insignificance value

df.train_rymodel_onlysig <- df.train_rymodel_onlysig[df.train_rymodel_onlysig$marital!="yes",]
result_tentative_trainrymodel_sig1 <- glm(formula = yact ~ age + balance + duration + campaign + previous + job + marital + education + housing + loan + poutcome, data=df.train_rymodel_onlysig, family = binomial)

df.test_rymodel_onlysig <- df.test_rymodel_onlysig[df.test_rymodel_onlysig$marital!="yes",]

summary(result_tentative_trainrymodel_sig1)

# removing the insignificant variable poutcomeother as it has the highest insignificance value

df.train_rymodel_onlysig <- df.train_rymodel_onlysig[df.train_rymodel_onlysig$poutcome!="other",]
result_tentative_trainrymodel_sig1 <- glm(formula = yact ~ age + balance + duration + campaign + previous + job + marital + education + housing + loan + poutcome, data=df.train_rymodel_onlysig, family = binomial)

df.test_rymodel_onlysig <- df.test_rymodel_onlysig[df.test_rymodel_onlysig$poutcome!="other",]

summary(result_tentative_trainrymodel_sig1)

# removing the insignificant variable educationunknown as it has the highest insignificance value

df.train_rymodel_onlysig <- df.train_rymodel_onlysig[df.train_rymodel_onlysig$education!="unknown",]
result_tentative_trainrymodel_sig1 <- glm(formula = yact ~ age + balance + duration + campaign + previous + job + marital + education + housing + loan + poutcome, data=df.train_rymodel_onlysig, family = binomial)

df.test_rymodel_onlysig <- df.test_rymodel_onlysig[df.test_rymodel_onlysig$education!="unknown",]

summary(result_tentative_trainrymodel_sig1)

# removing the insignificant variable jobunemployed as it has the highest insignificance value

df.train_rymodel_onlysig <- df.train_rymodel_onlysig[df.train_rymodel_onlysig$job!="unemployed",]
result_tentative_trainrymodel_sig1 <- glm(formula = yact ~ age + balance + duration + campaign + previous + job + marital + education + housing + loan + poutcome, data=df.train_rymodel_onlysig, family = binomial)

df.test_rymodel_onlysig <- df.test_rymodel_onlysig[df.test_rymodel_onlysig$job!="unemployed",]

summary(result_tentative_trainrymodel_sig1)

# removing the insignificant variable jobstudent as it has the highest insignificance value

df.train_rymodel_onlysig <- df.train_rymodel_onlysig[df.train_rymodel_onlysig$job!="student",]
result_tentative_trainrymodel_sig1 <- glm(formula = yact ~ age + balance + duration + campaign + previous + job + marital + education + housing + loan + poutcome, data=df.train_rymodel_onlysig, family = binomial)

df.test_rymodel_onlysig <- df.test_rymodel_onlysig[df.test_rymodel_onlysig$job!="student",]

summary(result_tentative_trainrymodel_sig1)

#no more insignificant variables left. All independent variables left behind are significant.

#Loading the final model into result_rymodel_sig1
result_rymodel_sig1 <- result_tentative_trainrymodel_sig1
class(result_rymodel_sig1)
print(result_rymodel_sig1)
plot(result_rymodel_sig1)

# Limitations of this model: Interactions are excluded; Linearity of independent variables is assumed #

fit_rymodel <- lm(formula <- yact ~ age + balance + duration + campaign + previous +
                   job + marital + education + housing + loan + poutcome, 
                 data=df.train_rymodel_onlysig)
vif(fit_rymodel)

# automated variable selection - Backward
backward_rymodel <- step(result_rymodel_sig1, direction = 'backward')
summary(backward_rymodel)

# training probabilities and roc
result_rymodel_probs <- df.train_rymodel_onlysig
nrow(result_rymodel_probs)
class(result_rymodel_probs)
#Using the model made to make predictions in the column named 'prob'
result_rymodel_probs$prob = predict(result_rymodel_sig1, type=c("response"))
q_rymodel <- roc(y ~ prob, data = result_rymodel_probs)
plot(q_rymodel)
auc(q_rymodel)

# how the categorical variables are distributed and are related with target outcome
CrossTable(df.train_rymodel_onlysig$job, df.train_rymodel_onlysig$y)
CrossTable(df.train_rymodel_onlysig$marital, df.train_rymodel_onlysig$y)
CrossTable(df.train_rymodel_onlysig$education, df.train_rymodel_onlysig$y)
CrossTable(df.train_rymodel_onlysig$default, df.train_rymodel_onlysig$y)
CrossTable(df.train_rymodel_onlysig$housing, df.train_rymodel_onlysig$y)
CrossTable(df.train_rymodel_onlysig$loan, df.train_rymodel_onlysig$y)
CrossTable(df.train_rymodel_onlysig$poutcome, df.train_rymodel_onlysig$y)

# numerical variable distribution
hist(df.train_rymodel_onlysig$age)
hist(df.train_rymodel_onlysig$balance)
hist(df.train_rymodel_onlysig$duration)
hist(df.train_rymodel_onlysig$campaign)
hist(df.train_rymodel_onlysig$previous)

# confusion matrix on ry-model
# to check the accuracy of the model made by removing all the insignificant variables
result_rymodel_probs$ypred = ifelse(result_rymodel_probs$prob>=.5,'pred_yes','pred_no')
table(result_rymodel_probs$ypred,result_rymodel_probs$y)

#probabilities on test set
df.test_rymodel_onlysig$prob = predict(result_rymodel_sig1, newdata = df.test_rymodel_onlysig, type=c("response"))

#confusion matrix on test set
df.test_rymodel_onlysig$ypred = ifelse(df.test_rymodel_onlysig$prob>=.5,'pred_yes','pred_no')
table(df.test_rymodel_onlysig$ypred,df.test_rymodel_onlysig$y)

# ks plot #
ks_plot(actuals=result_rymodel_probs$y, predictedScores=result_rymodel_probs$ypred)

# STEP 4: END

############### RY - code End #############


# DO IT YOURSELF ------------------------------------------------------------------
# Improve your model by removing insignificant variables
# Use automated variable selection to improve models
# check performance of test data
#-------------------------------------------------------------------------------------

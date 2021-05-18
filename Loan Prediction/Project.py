import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(0)

train = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
test = pd.read_csv('test_Y3wMUE5_7gLdaTN.csv')

#Evaluating missing values
print(train.isna().sum()) #dealling with missing values, in the future

sns.countplot(train['Loan_Status'])
plt.show()
print(train['Loan_Status'].value_counts())

train.boxplot(column='ApplicantIncome', by= ['Education'])
plt.show()

#It seems that people with highest income, tend to be graduated.

train.boxplot(column='ApplicantIncome', by = ['Loan_Status'])

#We should thought that people with highest income, should have highest chances of having a loan, but
#in this graphic we can see, that this is not that clear.

train.boxplot(column='LoanAmount', by = ['Loan_Status'])
#The lowes the loan amount, the highest the possibilities the loan to be accepted

plt.show()

### DATA WRANGLING ##
# After all the graphic information, we have to preprocess our data, in order to be able to work with it

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# to deal with missing values, Imputation is a good option. It is based on substitute the missing value
# for another value (usually the mean)

# Dependents variable have int and string values. To make the imputation, is neccessary that
# all the values are the same type

# THE PROCESS OF PREPROCCESSING THE DATA IS GONNA BE TEDIOUS, BECAUSE THE DATASET WAS FIRSTLY DIVIDED INTO
# TRAIN AND TEST, SO WE HAVE TO DO THE DATA WRANGLING TO TIMES (THE BEST WAY TO DO THAT IS TO PREPROCCESS THE
# DATA AND THEN DIVIDE IT IN TRAIN AND TEST)

#Replace is a good way to make numerical, some categorical varibales
train['Dependents'] = train['Dependents'].replace('3+', 3)
test['Dependents'] = test['Dependents'].replace('3+', 3)
train['Loan_Status'] = train['Loan_Status'].replace(['N', 'Y'], [0, 1])
train['Self_Employed'] = train['Self_Employed'].replace(['No', 'Yes', 'Not'], [0, 1, 0])
test['Self_Employed'] = test['Self_Employed'].replace(['No', 'Yes', 'Not'], [0, 1, 0])
train['Gender'] = train['Gender'].replace(['Male', 'Female'], [1, 0])
test['Gender'] = test['Gender'].replace(['Male', 'Female'], [1, 0])
train['Married'] = train['Married'].replace(['No', 'Yes'], [0, 1])
test['Married'] = test['Married'].replace(['No', 'Yes'], [0, 1])
train['Education'] = train['Education'].replace(['Not Graduate', 'Graduate'], [0, 1])
test['Education'] = test['Education'].replace(['Not Graduate', 'Graduate'], [0, 1])
train['Property_Area'] = train['Property_Area'].replace(['Urban', 'Semiurban', 'Rural'], [2, 1, 0])
test['Property_Area'] = test['Property_Area'].replace(['Urban', 'Semiurban', 'Rural'], [2, 1, 0])

imputer = SimpleImputer()

train['Gender']= imputer.fit_transform(train[['Gender']])
test['Gender']= imputer.fit_transform(test[['Gender']])
train['Dependents']= imputer.fit_transform(train[['Dependents']])
test['Dependents']= imputer.fit_transform(test[['Dependents']])
train['Self_Employed'] = imputer.fit_transform(train[['Self_Employed']])
test['Self_Employed'] = imputer.fit_transform(test[['Self_Employed']])
train['Loan_Amount_Term']= imputer.fit_transform(train[['Loan_Amount_Term']])
test['Loan_Amount_Term']= imputer.fit_transform(test[['Loan_Amount_Term']])
train['Credit_History']= imputer.fit_transform(train[['Credit_History']])
test['Credit_History']= imputer.fit_transform(test[['Credit_History']])
train['LoanAmount'] = imputer.fit_transform(train[['LoanAmount']])
test['LoanAmount'] = imputer.fit_transform(test[['LoanAmount']])
#as married column has only 3 missing values, eliminating them is a good idea
train = train.dropna(axis=0, subset=['Married'])
print('NA values in the train dataset:\n', train.isna().sum())
test = test.dropna(axis=0, subset=['Married'])
print('\n')
print('NA values in the test dataset:\n',test.isna().sum())

#There are no missing values, the data es preproccessed and everything is numerical, so now,
# we can start with the model creation

#The last part of the preprocessing is to detect and eliminate outliers, cause they can affect the
#results of the future models.
#so the only variables in which we can find outliers are ApplicantIncome, CoapplicantIncome and LoanAmount
#After starting with the models, we have to deal with outliers. TO do that we are gonna use Z-score
from scipy import stats

z_scores_train = np.abs(stats.zscore(train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']]))
z_scores_test = np.abs(stats.zscore(test[['ApplicantIncome', 'CoapplicantIncome']]))
filtered_entries = (z_scores_train < 3). all(axis=1)
filtered_entries_test = (z_scores_test < 3). all(axis=1)
train = train[filtered_entries]
test = test[filtered_entries_test]

#DATA WRANGLING COMPLETED#

#CORRRELATION MATRIX
#First we are gonna create a heatmap of correlation, to evaluate the most correlated variables
matrix = train.corr()
sns.heatmap(matrix, annot=True)
plt.show()

#as we can see in the heatmap, the most correlated variables are:
# --> LoanAmount and ApplicantIncome
# --> Credit_History and Loan_Status

## LOGISTIC REGRESSION ##
#the test dataset, doesn´t contain the test target, so we can´t test
#the accuracy of the model in that way. What we can do, is split the train
# dataset into train and validation, and measuring the accuracy using that
from sklearn.model_selection import train_test_split

x_train = train.drop(['Loan_Status', 'Loan_ID'], axis = 1)
y_train = train['Loan_Status']

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=10000)
model.fit(x_train, y_train)
prediction = model.predict(x_valid)
accuracy = accuracy_score(y_valid, prediction)
print('Logistic Regression:\n', accuracy) #the accuracy of the model is 0,79

# THERE ARE SOME OTHER VARIABLES, BASED ON THE ONES WE ALREADY HAVE, THAT CAN BE INCLUDED IN THE DATASET
# AND CAN AFFECT THE RESULTS
# Total income --> ApplicantIncome + CoapplicantIncome
# EMI --> LoanAmount/Loan_amount_term
# Balance Income --> Total income - (EMI*1000)

x_train['Total_Income'] = x_train['ApplicantIncome'] + x_train['CoapplicantIncome']
test['Total_Income'] = test['ApplicantIncome'] + test['CoapplicantIncome']
x_train['EMI'] = x_train['LoanAmount'] / x_train['Loan_Amount_Term']
test['EMI'] = test['LoanAmount'] / test['Loan_Amount_Term']
x_train['Balance_Income'] = x_train['Total_Income'] - (x_train['EMI'] * 1000)
test['Balance_Income'] = test['Total_Income'] - (test['EMI'] * 1000)

## OTHER MODELS ##
# LOGISTIC REGRESSION WITH THE NEW VARIABLES #
model2 = LogisticRegression(max_iter=10000)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)
model2.fit (x_train, y_train)
prediction2 = model2.predict(x_valid)
accuracy2 = accuracy_score(y_valid,prediction2)
print('Logistic Regression with new variables included:\n', accuracy2)
#the accuracy has decreased a little bit, after introducing the new variables, but it could mean
#that the model is now more real.

#  DECISSION TREE #
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
tree_prediction = tree.predict (x_valid)
accuracy3 = accuracy_score(y_valid, tree_prediction)
print('Decission tree accuracy:\n' ,accuracy3)

#a good way to improve the performance of a Decission Tree is to use AdaBoost, XGboost
# and Random Forest as other ensamble models

from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
adaboost = AdaBoostClassifier()
adaboost.fit(x_train, y_train)
prediction_adaboost = adaboost.predict(x_valid)
accuracy4 = accuracy_score(y_valid, prediction_adaboost)
print('Adaboost Classifier accuracy:\n', accuracy4)

xgboost = XGBClassifier()
xgboost.fit(x_train, y_train)
prediction_xgboost = xgboost.predict(x_valid)
accuracy5 = accuracy_score(y_valid, prediction_xgboost)
print('XGBoost Classifier accuracy:\n', accuracy5)

# RANDOM FOREST #
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state=5)
randomforest.fit(x_train, y_train)
prediction_forest = randomforest.predict(x_valid)
accuracy6 = accuracy_score(y_valid, prediction_forest)
print('Random Forest accuracy:\n', accuracy6)

#As we have seen, XGboost gave us the best performance with the basic model.
#Obviously we can tune de hyperparameters to maximize the efectiveness of the models
#We are going to tune the hyperparameters of the Random Forest as a example
from sklearn.model_selection import GridSearchCV

np.random.seed (0)

criterion = ['gini', 'entropy']
max_depth = [1,2,3,4,5,6,7,8,9,10]
n_estimators = [25,50,100,150]
max_features = ['auto', 'sqrt', 'log2']
hyperparameters = dict(criterion = criterion, max_depth = max_depth, n_estimators = n_estimators, max_features = max_features)

grid = GridSearchCV(randomforest, param_grid=hyperparameters, n_jobs=-1)
best_model = grid.fit (x_train, y_train)
print('Best criterion:\n', best_model.best_estimator_.get_params()['criterion'])
print('Best max_depth:\n', best_model.best_estimator_.get_params()['max_depth'])
print('Best n_estimators:\n', best_model.best_estimator_.get_params()['n_estimators'])
print('Best max_features:\n', best_model.best_estimator_.get_params()['max_features'])

#Finally we are going to create a new random forest, using these parameters

randomforest_optimized = RandomForestClassifier(criterion='entropy', max_depth=6, n_estimators=50, max_features='auto', n_jobs=-1)
randomforest_optimized.fit(x_train,y_train)
prediction_forest_optim = randomforest_optimized.predict(x_valid)
accuracy_optimized = accuracy_score(y_valid, prediction_forest_optim)
print('Random Forest optimized accuracy:\n', accuracy_optimized)




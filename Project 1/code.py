import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('max_columns', 15)

#loading the datasets
train = pd.read_csv('train_Df64byy.csv')
test = pd.read_csv('test_YCcRUnU.csv')

target = 'Response'
ID = test['ID']
#First look to the data
train.head()

#Some basic stats of the dataset
train.describe()

#We have to remove the ID variable, as it is not going to be part of the study
train = train.drop(['ID'], axis = 1)
test = test.drop(['ID'], axis = 1)
#SOME PLOTS TO ANALYSE THE DATA
#We need to divide the data into numerical and categorical variables, in order to
#do the proper plots

num_variables = train.select_dtypes(include = [np.number])
cat_variables = train.select_dtypes(include = [np.object])

#Boxplots for numerical variables
fig = plt.figure(figsize=(12,10))
for i in range(len(num_variables.columns)):
    fig.add_subplot(2, 4, i+1)
    sns.boxplot(y=num_variables.iloc[:,i])
plt.tight_layout()
plt.show()

#Countplots for categorical variables
fig2 = plt.figure(figsize = (12,15))
for i in range(len(cat_variables.columns)):
    fig2.add_subplot(2,3, i+1)
    sns.countplot(x=cat_variables.iloc[:,i])
plt.tight_layout()
plt.show()

#With this plots we can have an idea about the distribution of each variable

#DATA CLEANING
#Analysing the presence of missing values
(train.isna().sum())
(test.isna().sum())

#We have missing values in three diferent variables: Health Indicator, Holding_Policy_Type,
#and Holding_Policy_and Holding_Policy_Duration

#Dealing with missing values
#First, we are going to replace missing values by the most common values
train['Health Indicator'] = (train['Health Indicator'].fillna(train['Health Indicator'].mode()[0]))
test['Health Indicator'] = (test['Health Indicator'].fillna(test['Health Indicator'].mode()[0]))

#Before doing the same with Holding Policy variables, we are going to create a new variable
#that gets a 1 if the customer has a holding policy and a 0 if not.
train['Holding_policy'] = [1 if x>0 else 0 for x in (train['Holding_Policy_Type'])]
test['Holding_policy'] = [1 if x>0 else 0 for x in (test['Holding_Policy_Type'])]

train['Holding_Policy_Type'] = (train['Holding_Policy_Type'].fillna(train['Holding_Policy_Type'].mode()[0]))
test['Holding_Policy_Type'] = (test['Holding_Policy_Type'].fillna(test['Holding_Policy_Type'].mode()[0]))

#This variable contain numerical and categorical levels, so first, we have to replace
#the categorical level by a number, and then replace na values by the most common one
train['Holding_Policy_Duration'] = (train['Holding_Policy_Duration'].replace('14+', 15.0))
test['Holding_Policy_Duration'] = test['Holding_Policy_Duration'].replace('14+', 15.0)

train['Holding_Policy_Duration'] = (train['Holding_Policy_Duration'].fillna(train['Holding_Policy_Duration'].mode()[0])).astype('float64')
test['Holding_Policy_Duration'] = (test['Holding_Policy_Duration'].fillna(test['Holding_Policy_Duration'].mode()[0])).astype('float64')

#Correlation matrix
matrix = train.corr().abs()
sns.heatmap(matrix, annot= True)
plt.show()

#We can see correlation between Upper age and Lower age, so we have to remove one of them
#First we are going to create a new variable, called dif_age, which is the difference
#between upper and lower age
train['Dif_age'] = train['Upper_Age'] - train['Lower_Age']
test['Dif_age'] = test['Upper_Age'] - test['Lower_Age']

#Now we can remove Upper or Lower age, cause they are correlated and also included in this
#new variable. Upper Age is going to be dropped, cause it is also correlated with Reco_Policy_Premium
train = train.drop(['Upper_Age'], axis =1) #try then dropping lower age
test = test.drop(['Upper_Age'], axis =1)

#LABEL ENCODING
#There are some variables that need to be encoded
from sklearn.preprocessing import LabelEncoder
#This encoding is for variables whose levels dont have a hierarchical order, they are independent
train = pd.get_dummies(train, columns = ['City_Code','Accomodation_Type', 'Reco_Insurance_Type', 'Is_Spouse'])
test = pd.get_dummies(test, columns = ['City_Code','Accomodation_Type', 'Reco_Insurance_Type', 'Is_Spouse'])

#On the other hand, label encoder is used for variables whose levels need a hierarchical order, which means
#that distances between levels are important (the model needs to know that)
encoder = LabelEncoder()
train['Health Indicator'] = encoder.fit_transform(train['Health Indicator'])
test['Health Indicator'] = encoder.fit_transform(test['Health Indicator'])

train['Holding_Policy_Type'] = encoder.fit_transform(train['Holding_Policy_Type'])
test['Holding_Policy_Type'] = encoder.fit_transform(test['Holding_Policy_Type'])

#MODEL
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV

X_train = train.drop(['Region_Code', 'Response'], axis = 1)
Y_train = train['Response']
X_test = test.drop(['Region_Code'], axis = 1)

#LIGHT GBM
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2)
weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
weights = dict(enumerate(weights))

#MODEL
#LIGHTGBM
from sklearn.model_selection import RepeatedStratifiedKFold
model = LGBMClassifier(class_weight=weights, metric = 'roc_auc_score',
                       learning_rate=0.15, max_depth=14, feature_fraction =0.9,
                       num_leaves=2^14)

learning_rate = [0.01, 0.02, 0.05, 0.1, 0.12, 0.15,0.2]
max_depth = range (1,15)
feature_fraction = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
subsample = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
params = dict(learning_rate=learning_rate, max_depth=max_depth, feature_fraction=feature_fraction,
              subsample=subsample)

search = RandomizedSearchCV(model, param_distributions=params, cv=cv, n_jobs=-1, n_iter=10)
search.fit(x_train, y_train)
print("Best: %f using %s" % (search.best_score_, search.best_params_))

model.fit(X_train, Y_train)
prediction = model.predict(x_val)
accuracy = roc_auc_score(prediction, y_val)
print(accuracy)

prediction_bien = model.predict(X_test)

complete = pd.DataFrame({'ID': ID, target: prediction_bien})

complete.to_csv(r'E:\MACHINE LEARNING\Python projects\jobathon\lightgbm_bien.csv', index=False)

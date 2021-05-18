import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('diabetes.csv')
target = data['Outcome']
features = data.drop(['Outcome'], axis = 1)

(data.groupby('Outcome').size())
(data['Age'].min())
(data['Age'].max())
#this dataset contains 500 persons without diabetes, and 268 with diabetes

bins = [21, 30, 40, 50, 60, 70, 81]
labels = ['21-30', '31-40', '41-50', '51-60', '61-70', '71-81']
features['Age']= pd.cut(features.Age, bins, labels= labels, include_lowest=True)

sns.countplot(x = data['Outcome'], hue = features['Age'])
plt.show()

#in this plot we can see the people by age, with diabetes, and withouth it.

#IS KNN THE BEST MODEL? ##
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

features = data.drop(['Outcome'], axis = 1)

#we introduce again the features variable, because in the proccess of creating the plots, we have modified the feature
# 'age' , and we dont want it modified to make the model.
model = KNeighborsClassifier(n_neighbors=5)

#these method will help us to select the best number of neighbors
possible_neighbors = [{"n_neighbors": [5,10,14,15,16,17,18,19,20]}]
classifier = GridSearchCV(model, possible_neighbors, cv=5)

classifier.fit(features, target)
print(classifier.best_estimator_)
#14 neighbors is the best number of neighbors, so know, we can adjust the model with 14 neighbors

x_train,x_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
model = KNeighborsClassifier(n_neighbors=17, n_jobs=-1)
model.fit(x_train, y_train)
print('Knn')
print('Accuracy of the model in the training set:', model.score(x_train, y_train)) #accuracy of the model in the training set
print('Accuracy of the model in the test set:', model.score(x_test, y_test)) #accuracy of the model in the test set

#IS DECISSION TREE THE BEST MODEL?
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0)
tree.fit(x_train,y_train)
print('\nDecision Tree')
print('Accuracy of the model in the training set:', tree.score(x_train, y_train)) #accuracy of the model in the training set
print('Accuracy of the model in the test set:', tree.score(x_test, y_test)) #accuracy of the model in the test set

#As we can see, the performance of the decision tree is worse (in the test set) than the Knn algorithm, but in the training set
# is 1. This is caused by overftting, so we wave to make a pre-prunning to the tree

# a way of doing that is limiting the max_depth of the tree.

tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(x_train, y_train)
print('\nDecision Tree pruned')
print('Accuracy of the model in the training set:', tree.score(x_train, y_train)) #accuracy of the model in the training set
print('Accuracy of the model in the test set:', tree.score(x_test, y_test)) #accuracy of the model in the test set
print('\n')
#with a max_depth of 3, the decision tree makes its best performance

for i in range(features.columns.size):
    print(features.columns[i], tree.feature_importances_[i])

#as we can see the most important features in order to provoke diabetes are glucose, BMI and Age

#now we can make predictions introducing a new observation

#A randomly created observation to evaluate the model
new_obs = [[0,105,70,18,0,25.0,0.147,25]]
print(tree.predict(new_obs))



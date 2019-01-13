

# Data Representation in scikit learn

import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()

#features
X_iris = iris.drop('species', axis = 1)
X_iris.shape

#label
Y_iris = iris['species']
Y_iris.shape

#using Guassian Naive Bayes to learn

from sklearn.naive_bayes import GaussianNB   #1. choose model class
model = GaussianNB()    #2. instantiate model
model.fit(X_iris, Y_iris)
Y_model = model.predict(X_iris)

#The labels are strings so to use mean_squared_error we need to convert them to float (or int)
#We use preprocessing label Encoder
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

le.fit(Y_iris)
Y_iris_num =  le.transform(Y_iris)

le.fit(Y_model)
Y_model_num =  le.transform(Y_model)

error = mean_squared_error(Y_iris_num, Y_model_num)
print(error)

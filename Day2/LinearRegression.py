#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
filename='height-weight.csv'
raw_data = open(filename,'r')
data = numpy.loadtxt(raw_data, delimiter=',')
data.shape


# In[2]:


data[:10]  #showing the first 10 rows


# In[4]:


#importing the plotting library
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# We separate out the independent variable height into X
#and dependent variable weight into Y
X = data[:,0]
Y = data[:,1]

#plotting the first 20 data
X_20 = X[:20]
Y_20 = Y[:20]

plt.scatter(X_20, Y_20, color = 'red', s=30)   #s= area of marker


# In[5]:


#as the plot shows when the height increases the weight increases as well

#Split the data into training/testing sets
X_train = X[:4500]   #the first 4500 rows of height
X_test = X[4500:]

#Split the target into training/Testing sets
Y_train = Y[:4500]
Y_test = Y[4500:]

print(X_train.shape)
print(Y_train.shape)

print(X_test.shape)
print(Y_test.shape)


# In[6]:


#We need to convert it to an array
X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)

#It means put the second dimention to 1 and infer the first dimention from the length of the array
#One shape dimension can be -1. In this case, the value is inferred from the length of the array
#and remaining dimensions
X_train.shape


# In[7]:


#Create Linear regression object
regr = linear_model.LinearRegression()
#Train the model using the training set
regr.fit(X_train, Y_train)


# In[11]:


#Make predictions using teh testing set
Y_pred = regr.predict(X_test)

#The coefficients
print('Coefficients:\n', regr.coef_)
print(mean_squared_error(Y_test, Y_pred))


# In[10]:


plt.scatter(X_test, Y_test, color='black')
plt.plot(X_test, Y_pred, color = 'blue', linewidth = 3)
plt.xticks(())
plt.yticks(())
plt.show

#ticks : array_like
#A list of positions at which ticks should be placed. You can pass an empty list to disable xticks.


# In[ ]:





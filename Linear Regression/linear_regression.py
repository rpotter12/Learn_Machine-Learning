# Profit estimation of companies with linear regression

# importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns

# load dataset
companies = pd.read_csv('1000_Companies.csv')
print(companies.head())

# extracting independent and dependent variables
# here we are taking all column except last column and first 5 rows
X = companies.iloc[:, :-1].values
y = companies.iloc[:, 4].values

# visualizing the data before processing
print(sns.heatmap(companies.corr()))

# encoding categorical data
# our model can't train using categorial data like standards in .csv file. from this we convert data in numbers.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categories = 'auto')
X = onehotencoder.fit_transform(X).toarray()
print(X[0])

# avoiding dummy data trap
X = X[:,1:]

# splitting dataset into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

# fitting model to training set
# here we train our Linear Regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# predicting the test dataset
y_pred = lin_reg.predict(X_test)

# finding coefficients and intercepts
# here it prints the coefficient and intercept of the line. y=mx+c is the standard equation. here it do the math part
print(lin_reg.coef_)
print(lin_reg.intercept_)

# here it calculate the accuracy of the model
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

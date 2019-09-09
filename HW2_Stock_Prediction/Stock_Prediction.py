# Step 1: Download historical data from Yahoo Finance and save the data in a .csv file
### Date of extraction: 09 Sep 2019
### Company chosen: COF (Capital One Finance)
### Time period of historical data: 5 years, starting from 08 Sep 2014
### File name: COF.csv
#import necessary modules
import pandas as pd
data = pd.read_csv('COF.csv')

# Step 2: Use Scikit learn to try out 3 different types of regression models to predict the price of that stock for a future date
### 1. Processing
#import necessary modules
import math
import numpy as np
from sklearn import preprocessing

# Create new features
dfreg = data.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (data['High'] - data["Low"]) / data['Close'] * 100.0
dfreg['PCT_change'] = (data['Close'] - data['Open']) / data['Open'] * 100.0

# Drop missing value
dfreg.fillna(value=-99999, inplace=True)

# We want to separate 5 percent of the data to forecast
forecast_out = int(math.ceil(0.05 * len(dfreg)))

# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))

# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)

# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_train = X[:-forecast_out] # Exclude rows from bottom
X_future = X[-forecast_out:] # Filter rows from bottom

# Separate label and identify it as y
y = np.array(dfreg['label'])
y_train = y[:-forecast_out]

### 2. Models considered: OLS regression, Lasso Regression, Ridge regression
#import necessary modules
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

##### Model 1: OLS Regression
lin_reg = linear_model.LinearRegression()
lin_reg.fit(X_train, y_train)
lin_reg_y_predict = lin_reg.predict(X_train) # predicting price using the model given the input data
# in-sample performance
print(mean_squared_error(y_train, lin_reg_y_predict)) # 58.42
print(r2_score(y_train, lin_reg_y_predict)) # 0.5107

##### Model 2: Lasso Regression
lasso_reg = linear_model.Lasso(alpha=0.1) 
lasso_reg.fit(X_train, y_train) 
lasso_reg_y_predict = lasso_reg.predict(X_train) # predicting price using the model given the input data
# in-sample performance
print(mean_squared_error(y_train, lasso_reg_y_predict)) # 58.47
print(r2_score(y_train, lasso_reg_y_predict)) # 0.5103

##### Model 3: Ridge Regression
ridge_reg = linear_model.Ridge(alpha=100) # 100 is big, but still using it to see the difference in performance between OLS and ridge; cross-validation can be done to get alpha but not doing it here
ridge_reg.fit(X_train, y_train) 
ridge_reg_y_predict = ridge_reg.predict(X_train) # predicting price using the model given the input data
# in-sample performance
print(mean_squared_error(y_train, ridge_reg_y_predict)) # 58.83
print(r2_score(y_train, ridge_reg_y_predict)) # 0.5073

# Among all the three models, the in-sample performance measures (mean square error and R square) are better for OLS regression model. Hence, that is our final model for prediction.
# Predcit stock price at a future date. In this case, on the test data
lin_reg_y_future = lin_reg.predict(X_future)
print(lin_reg_y_future)

# Step 3: Visualize your result using matplotlib or another plotting library of your choice
#import necessary modules
%matplotlib inline
import matplotlib.pyplot as plt

# Separating original data into train and prediction sets
data_train = data[:-forecast_out]
data_predict = data[-forecast_out:]

# Add new column to above split datasets
data_train['Forecast'] = np.nan # Nothing predicted. Hence NaN.
data_predict['Forecast'] = lin_reg_y_future.tolist() # Adding predicted values

# Joining the split datasets
data_full = data_train.append(data_predict)

# Plotting stock price over time
plt.plot(dfreg['label']) # Used as target during training
plt.plot(data_full['Forecast']) # Predicted by model



from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

import yfinance as yf

import datetime as dt
current_date = dt.datetime.now()

Df = yf.download('GLD', '2011-05-01', current_date, auto_adjust=True)
Df = Df[['Close']]
Df = Df.dropna()

Df.Close.plot(figsize=(10, 7),color='r')
plt.ylabel("Gold ETF Prices")
plt.title("Gold ETF Price Series")
plt.show()

# Define explanatory variables
Df['S_3'] = Df['Close'].rolling(window=3).mean() 
Df['S_9'] = Df['Close'].rolling(window=9).mean() 
Df['next_day_price'] = Df['Close'].shift(-1)

#dropping all Nan values
Df = Df.dropna()
X = Df[['S_3', 'S_9']]

# Define dependent variable
y = Df['next_day_price']

#splitting into test and train dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Create a linear regression model
linear = LinearRegression().fit(X_train, y_train)

#explaining the equation used for our model
print("Linear Regression model")
print("Price (y) = %.2f * 3 Day Moving Average (x1) \
+ %.2f * 9 Day Moving Average (x2) \
+ %.2f (constant)" % (linear.coef_[0], linear.coef_[1], linear.intercept_))

# Predicting the Gold ETF prices
predicted_price = linear.predict(X_test)
predicted_price = pd.DataFrame(
    predicted_price, index=y_test.index, columns=['price'])
predicted_price.plot(figsize=(10, 7))
y_test.plot()
plt.legend(['predicted_price', 'actual_price'])
plt.ylabel("Gold ETF Price")
plt.show()

# R square to check our error rate
r2_score = linear.score(X_test, y_test)*100
float("{0:.2f}".format(r2_score))

# Forecast the price
Df['predicted_gold_price'] = linear.predict(Df[['S_3', 'S_9']])
Df['signal'] = np.where(Df.predicted_gold_price.shift(1) < Df.predicted_gold_price,"Buy","No Position")

# Print the forecast
Df.tail(1)[['signal','predicted_gold_price']].T

# Recurrent Neural Network


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Importing the training set
dataset = pd.read_csv('Datasets/Cell_000231/train_Cell000231.csv')
training_data = dataset.drop(["DateTime"], axis=1)

# Feature Scaling

scaler = MinMaxScaler()
training_dataset = scaler.fit_transform(training_data)

# Creating a data structure with 24 timesteps and 1 output
X_train = []
y_train = []
for i in range(24, training_dataset.shape[0]):
    X_train.append(training_dataset[i - 24:i, 0])
    y_train.append(training_dataset[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Initialising the RNN
regressior = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressior.add(LSTM(units=60, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressior.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressior.add(LSTM(units=60, return_sequences=True))
regressior.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressior.add(LSTM(units=80, return_sequences=True))
regressior.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressior.add(LSTM(units=120))
regressior.add(Dropout(0.2))

# Adding the output layer
regressior.add(Dense(units=1))

# Compiling the RNN
regressior.compile(optimizer='nadam', loss='mean_squared_error')

# Fitting the RNN to the Training set
history = regressior.fit(X_train, y_train, epochs=120, batch_size=39)

# Part 3 - Making the predictions and visualising the results

# Getting the real traffic values of Cell_000112
dataset_test = pd.read_csv('Datasets/Cell_000231/test_Cell000231.csv')
date_only = dataset_test.drop(['Traffic'], axis=1)
past_24_hours = dataset.tail(24)
df = past_24_hours.append(dataset_test, ignore_index=True)
df = df.drop(['DateTime'], axis=1)

# Getting the predicted traffic values of Cell_000112
inputs = scaler.fit_transform(df)
X_test = []
y_test_scaled = []
for i in range(24, inputs.shape[0]):
    X_test.append(inputs[i - 24:i])
    y_test_scaled.append(inputs[i, 0])
X_test, y_test_scaled = np.array(X_test), np.array(y_test_scaled)
predicts_scaled = regressior.predict(X_test)
scale_value = scaler.scale_
scale = 1 / scale_value
predicts = predicts_scaled * scale
y_test = y_test_scaled * scale
y_test = np.reshape(y_test, (y_test.shape[0], 1))



# ========================== ARIMA =======================================================================

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams


rcParams['figure.figsize']  = 16, 9
dataset = pd.read_csv('Datasets/Cell_000231/test_Cell000231.csv')
dataset.set_index('DateTime', inplace=True)
rolmean = dataset.rolling(window=24).mean()
rolstd = dataset.rolling(window=24).std()
print(rolmean, rolstd)
orig = plt.plot(dataset.values, color = 'blue', label='Original')
mean = plt.plot(rolmean.values, color='red', label='Rolling Mean')
std = plt.plot(rolstd.values, color='black', label='Rolling STD')
plt.legend(loc='best')
plt.show(block='False')

# Testing For Stationarity

from statsmodels.tsa.stattools import adfuller

test_result=adfuller(dataset['Traffic'])
#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(traffic):
    result=adfuller(traffic)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


adfuller_test(dataset['Traffic'])

# Autocorrelation

from pandas.plotting import autocorrelation_plot
plt.figure(figsize=(16, 9))
autocorrelation_plot(dataset['Traffic'])
plt.show()

import statsmodels.api as sm
# from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dataset['Traffic'],lags=10,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dataset['Traffic'],lags=10,ax=ax2)
plt.show()

from statsmodels.tsa.arima_model import ARIMA

model=ARIMA(dataset['Traffic'],order=(5,0,2))
results=model.fit(disp=-1)
final = results.fittedvalues

#Plot

plt.plot(dataset["Traffic"].values, color='blue', label='Real LTE Traffic')
plt.plot(final.values, color='red', label='ARIMA')
plt.plot(predicts, color='green', label='LSTM')
plt.title('Model validation for LSTM Based LTE Traffic Predictor')
plt.xlabel('Input from Test CSV')
plt.ylabel('Traffic')
plt.legend()
plt.show()
plt.show()

#  ARIMA and LSTM Error

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math

mse1 = mean_squared_error(y_test, predicts)
r21 = r2_score(y_test, predicts)
mae1 = mean_absolute_error(y_test, predicts)
print("LSTM Mean Squared Error: ", math.sqrt(mse1))
print("LSTM R2 Squared Error: ", r21)
print("LSTM Mean Absolute Error: ", mae1)

r2 = r2_score(dataset["Traffic"].values, results.fittedvalues)
mse = mean_squared_error(dataset["Traffic"].values, results.fittedvalues)
mae = mean_absolute_error(dataset["Traffic"].values, results.fittedvalues)
print("ARIMA R2 Squared Error: ", r2)
print("ARIMA Mean Squared Error: ", math.sqrt(mse))
print("ARIMA Mean Absolute Error: ", mae)


import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from scipy.ndimage.interpolation import shift
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
import matplotlib
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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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
# plt.plot(predicts, color='green', label='LSTM')
plt.title('Model validation for LSTM Based LTE Traffic Predictor')
plt.xlabel('Input from Test CSV')
plt.ylabel('Traffic')
plt.legend()
plt.show()
# plt.plot(history.history['loss'])
plt.show()

#  ARIMA Error

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math

r2 = r2_score(dataset["Traffic"].values, results.fittedvalues)
mse = mean_squared_error(dataset["Traffic"].values, results.fittedvalues)
print("ARIMA R2 Squared Error: ", r2)
print("ARIMA Mean Squared Error: ", math.sqrt(mse))
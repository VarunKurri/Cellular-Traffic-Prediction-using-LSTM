# Recurrent Neural Network


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Importing the training set
dataset = pd.read_csv('train_Cell000111.csv')
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

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
history = regressior.fit(X_train, y_train, epochs=120, batch_size=32)

# Part 3 - Making the predictions and visualising the results

# Getting the real traffic values of Cell Tower
dataset_test = pd.read_csv('test_Cell000111.csv')
date_only = dataset_test.drop(['Traffic'], axis=1)
past_24_hours = dataset.tail(24)
df = past_24_hours.append(dataset_test, ignore_index=True)
df = df.drop(['DateTime'], axis=1)

# Getting the predicted traffic values of Cell Tower
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
true_test = pd.DataFrame({'ActualTraffic': y_test[:, 0]})
true_predict = pd.DataFrame({'PredictedTraffic': predicts[:, 0]})
predicted_plotly = date_only.join(true_predict)
true_plotly = date_only.join(true_test)
result = true_plotly.join(true_predict)

# Visualising the results
plt.figure(figsize=(16, 9))
plt.plot(y_test, color='red', label='Real LTE Traffic')
plt.plot(predicts, color='blue', label='Predicted LTE Traffic')
plt.title('Model validation for LSTM Based LTE Traffic Predictor')
plt.xlabel('Input from Test CSV')
plt.ylabel('Traffic')
plt.legend()
plt.show()
plt.figure(figsize=(16, 9))
plt.plot(history.history['loss'])
plt.show()

import plotly.offline as pyo
import plotly.graph_objs as go

trace0 = go.Scatter(x=result.DateTime, y=result.ActualTraffic, mode='lines', name='Actual Traffic')
trace1 = go.Scatter(x=result.DateTime, y=result.PredictedTraffic, mode='lines', name='Predicted Traffic')
data = [trace0, trace1]
layout = go.Layout(title='Actual vs. Predicted Traffic', xaxis_title="Date & Time", yaxis_title="Traffic in Mb/s")
figure = go.Figure(data=data, layout=layout)
pyo.plot(figure)
thresh = 0.5
y_pred = (predicts_scaled >= 0.5).astype('int')
y_act = (y_test_scaled >= 0.5).astype('int')
cm = confusion_matrix(y_act, y_pred)
class_names = ['True', 'False']
fig, ax = plot_confusion_matrix(conf_mat=cm, show_absolute=True, show_normed=True, colorbar=True, class_names=class_names)
plt.show()
tn, fp, fn, tp = confusion_matrix(y_act, y_pred).ravel()
print("True Negatives: ", tn)
print("False Positives: ", fp)
print("False Negatives: ", fn)
print("True Positives: ", tp)
Accuracy = (tn + tp) * 100 / (tp + tn + fp + fn)
print("Accuracy = ", (Accuracy))

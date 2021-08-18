"""
Fully-Connected Neural Network for a Regression task.

metrics (tensorflow.keras.metrics):
mean_squared_error
mean_squared_logarithmic_error
mean_absolute_error
mean_absolute_percentage_error
cosine_proximity
"""
import numpy as np

from data_tools.data import RegressionDataSets
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
import matplotlib.pyplot as plt

####################################

# Part 1 - Data Preprocessing

dataset = RegressionDataSets()
dataset.get_Combined_Cycle_Power_Plant()

# full Feature Scaling
#   For all independent variables (even dummy variables).
#   This is absolutely compulsory for deep learning.
sc = StandardScaler()
X_train_sc = sc.fit_transform(dataset.X_train)
X_test_sc = sc.transform(dataset.X_test)

####################################

# Part 2 - FC-NN

# Building:
#   Layers: Input layer, (Hidden layer) x #, Output Layer
classifier = Sequential()  # model, regressor
classifier.add(Dense(units=6, activation='relu', name='hidden_1'))
classifier.add(Dense(units=6, activation='relu', name='hidden_2'))
classifier.add(Dense(units=1, name='output'))

# Compiling
metrics=['mse', 'msle', 'mae', 'mape', 'cosine_similarity']
classifier.compile(optimizer='adam', loss='mse', metrics=metrics)

# Training
history = classifier.fit(X_train_sc, dataset.y_train,
                         validation_data=(X_test_sc, dataset.y_test),
                         batch_size=32, epochs=50, verbose=2)

# plot metrics
for metric in metrics:
    plt.title(metric)
    plt.plot(history.history[metric])
    plt.show()

# Making predictions
y_pred = classifier.predict(X_test_sc)

# evaluating the model
r2 = r2_score(dataset.y_test, y_pred)
explained_variance = explained_variance_score(dataset.y_test, y_pred)
mse = mean_squared_error(dataset.y_test, y_pred)
print(f'R^2 score: {r2:.5f}')
print(f'Explained Variance score: {explained_variance:.2f}')
print(f'MSE: {mse:.2f}')

####################################

# Part 3 - Making a single prediction
#   Predicting the result of a single observation:
X_new = np.array([[20, 50, 1000, 80]])
X_new = sc.transform(X_new.astype(float))
y_pred_new = classifier.predict(X_new)
print('Single prediction:', y_pred_new[0, 0])

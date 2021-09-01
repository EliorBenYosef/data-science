"""
Fully-Connected Neural Network for a Binary Classification task
(a Business Problem - geodemographic segmentation model).

metrics (tensorflow.keras.metrics):
accuracy, acc
    binary_accuracy
    categorical_accuracy
sparse_categorical_accuracy
top_k_categorical_accuracy (requires you specify a k parameter)
sparse_top_k_categorical_accuracy (requires you specify a k parameter)
"""

from data_tools.data import ClassificationDataSets
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

####################################

# Part 1 - Data Pre-Processing

dataset = ClassificationDataSets()
dataset.get_churn_modelling()

# full Feature Scaling
#   For all independent variables (even dummy variables).
#   This is absolutely compulsory for deep learning.
sc = StandardScaler()
dataset.X_train_sc = sc.fit_transform(dataset.X_train_sc)
dataset.X_test_sc = sc.transform(dataset.X_test_sc)

####################################

# Part 2 - FC-NN

# Building:
#   Layers: Input layer, (Hidden layer) x #, Output Layer
classifier = Sequential()
classifier.add(Dense(units=6, activation='relu', name='hidden_1'))
classifier.add(Dense(units=6, activation='relu', name='hidden_2'))
classifier.add(Dense(units=1, activation='sigmoid', name='output'))
# classifier.add(Dense(units=n_clss, activation='softmax', name='output'))

# Compiling
metrics=['accuracy']
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
# classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

# Training
history = classifier.fit(dataset.X_train_sc, dataset.y_train,
                         validation_data=(dataset.X_test_sc, dataset.y_test),
                         batch_size=32, epochs=50, verbose=2)  # epochs=100

# plot metrics
for metric in metrics:
    plt.plot(history.history[metric])
plt.show()

# Making predictions
y_pred = classifier.predict(dataset.X_test_sc)
y_pred = (y_pred > 0.5).astype(np.int)

# evaluating the model
accuracy = accuracy_score(dataset.y_test, y_pred)
c_matrix = confusion_matrix(dataset.y_test, y_pred)
clss_report = classification_report(dataset.y_test, y_pred)
print(f'Accuracy score: {accuracy:.2f}')
print(f'Confusion Matrix: \n{c_matrix}')
print(f'classification report: \n{clss_report}')

####################################

# Part 3 - Making a single prediction
#   Predicting the result of a single observation:
#   CreditScore, Country, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
#   600, France, Male, 40 years old, 3 years, $60000, 2, 1 (Yes), 1 (Yes), $50000
X_new = [[600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]]
for t in dataset.transformers:
    if isinstance(t.transformers[0][1], StandardScaler):
        X_new = t.transform(X_new.astype(float))
    else:
        X_new = t.transform(X_new)
X_new = sc.transform(X_new.astype(float))
y_pred_new = classifier.predict(X_new)
y_pred_new = (y_pred_new > 0.5).astype(np.int)
print('Single prediction:', y_pred_new[0, 0])

"""
for binary classification.

https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-3/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from machine_learning.utils import sigmoid_activation, log_loss, single_gradient_step


regularized = True  # regularized = with polynomial features


if not regularized:
    path = '../../../../datasets/per_field/sl/clss/logreg_simple.txt'
    names = ['Exam 1', 'Exam 2', 'Admitted']
else:
    path = '../../../../datasets/per_field/sl/clss/logreg_simple_regularized.txt'
    names = ['Test 1', 'Test 2', 'Accepted']
df = pd.read_csv(path, header=None, names=names)


# Let's start by examining the data (exploratory analysis stage):
print(df.head(), '\n')
print(df.describe(), '\n')

positive = df[df[names[2]].isin([1])]
negative = df[df[names[2]].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive[names[0]], positive[names[1]], s=50, c='b', marker='o', label=names[2])
ax.scatter(negative[names[0]], negative[names[1]], s=50, c='r', marker='x', label='Not ' + names[2])  # Rejected
ax.legend()
# ax.legend(loc='best', shadow=False, scatterpoints=1)
ax.set_xlabel(names[0] + ' Score')
ax.set_ylabel(names[1] + ' Score')
plt.show()


# We can test the cost function to make sure it’s working, but first we need to do some setup.
if not regularized:
    # add a ones column - this makes the matrix multiplication work out easier
    df.insert(0, 'Ones', 1)
else:  # exercise_type == TWO_CLASSES_REGULARIZED
    # when there is no linear decision boundary that will perform well on this data.
    #   One way to deal with this using a linear technique like logistic regression is to construct features that are
    #   derived from polynomials of the original features.
    #   We can try creating a bunch of polynomial features to feed into the classifier.
    degree = 5
    x1 = df[names[0]]
    x2 = df[names[1]]

    df.insert(3, 'Ones', 1)

    for i in range(1, degree):
        for j in range(0, i):
            df['F' + str(i) + str(j)] = np.power(x1, i - j) * np.power(x2, j)

    df.drop(names[0], axis=1, inplace=True)
    df.drop(names[1], axis=1, inplace=True)

    print(df.head(), '\n')


# set X (training data) and y (target variable)
cols = df.shape[1]
if not regularized:
    X = df.iloc[:, 0:cols - 1]
    y = df.iloc[:, cols - 1:cols]
else:  # remember from above that we moved the label to column 0
    X = df.iloc[:, 1:cols]
    y = df.iloc[:, 0:1]


# convert to numpy arrays and initialize the parameter array theta (model parameters)
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(cols - 1)
print('X.shape, y.shape, theta.shape:', X.shape, y.shape, theta.shape)

# Now let’s compute the cost for our initial solution given zeros for the model parameters, here represented as “theta”.
if not regularized:
    print('initial cost =', log_loss(theta, X, y))
else:
    print('initial cost =', log_loss(theta, X, y, 1))


# In the exercise, an Octave function called "fminunc" is used to optimize the parameters
#   given functions to compute the cost and the gradients.
# Since we're using Python, we can use SciPy's optimization API to do the same thing.
if not regularized:
    theta_opt = opt.fmin_tnc(func=log_loss, x0=theta, fprime=single_gradient_step, args=(X, y))
    print('final cost =', log_loss(theta_opt[0], X, y))
else:
    learningRate = 1
    theta_opt = opt.fmin_tnc(func=log_loss, x0=theta, fprime=single_gradient_step, args=(X, y, learningRate))
    print('final cost =', log_loss(theta_opt[0], X, y, learningRate))


def predict(theta, X):
    """
    a function to output predictions for a dataset X using our learned parameters theta.
    We can then use this function to score the training accuracy of our classifier.
    """
    probability = sigmoid_activation(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


# theta_min = np.matrix(theta_opt[0])
theta_min = np.array(theta_opt[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))

# Our logistic regression classifer correctly predicted if a student was admitted or not 89% of the time. Not bad! Keep in mind that this is training set accuracy though. We didn't keep a hold-out set or use cross-validation to get a true approximation of the accuracy so this number is likely higher than its true performance (this topic is covered in a later exercise).
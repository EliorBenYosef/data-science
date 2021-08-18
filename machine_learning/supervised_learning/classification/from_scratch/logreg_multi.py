"""
for multi-class classification.

https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-4/
"""

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from machine_learning.utils import sigmoid_activation, softmax_loss, single_gradient_step


def one_vs_all(X, y, num_labels, learning_rate):
    """
    the classifier. since logistic regression is only able to distiguish between 2 classes at a time,
        we need a strategy to deal with the multi-class scenario.
    In this exercise we're tasked with implementing a one-vs-all classification approach,
        where a label with k different classes results in k classifiers,
        each one deciding between "class i" and "not class i" (i.e. any class other than i).
    We're going to wrap the classifier training up in one function that computes the final weights
        for each of the 10 classifiers and returns the weights as a k X (n + 1) array, where n is the number of parameters.
    """
    rows = X.shape[0]
    params = X.shape[1]

    # k X (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))

    # insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # minimize the objective function
        fmin = minimize(fun=softmax_loss, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=single_gradient_step)
        all_theta[i - 1, :] = fmin.x

    return all_theta


def predict_all(X, all_theta):
    rows = X.shape[0]
    # params = X.shape[1]
    # num_labels = all_theta.shape[0]

    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # # convert to matrices
    # X = np.matrix(X)
    # all_theta = np.matrix(all_theta)

    h = sigmoid_activation(X * all_theta.T)  # compute the class probability for each class on each training instance
    h_argmax = np.argmax(h, axis=1)  # create array of the index with the maximum probability
    h_argmax = h_argmax + 1  # because our array was zero-indexed we need to add one for the true label prediction
    return h_argmax


data = loadmat('../../../../datasets/per_type/matlab/digits.mat')

X = data['X']
y = data['y']
all_theta = one_vs_all(X, y, 10, 1)

# One of the more challenging parts of implementing vectorized code is getting all of the matrix interactions
#   written correctly, so I find it useful to do some sanity checks by looking at the shapes of the arrays/matrices
#   I'm working with and convincing myself that they're sensible.
#   Let's look at some of the data structures used in the above function (one_vs_all):
# rows = X.shape[0]
# X = np.insert(X, 0, values=np.ones(rows), axis=1)
# y_0 = np.array([1 if label == 0 else 0 for label in y])
# y_0 = np.reshape(y_0, (rows, 1))
# print('X.shape, y_0.shape, all_theta.shape:', X.shape, y_0.shape, all_theta.shape)
# print('np.unique(y):', np.unique(y))

y_pred = predict_all(X, all_theta)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print('accuracy = {0}%'.format(accuracy * 100))  # accuracy = 74.6%

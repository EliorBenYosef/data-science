# https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-5/

from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize


# Implementing a rudimentary feed-forward neural network with backpropagation
#   and used it to classify images of handwritten digits.
#   the whole thing basically boils down to a series of matrix multiplications.
#   this is by far the most efficient way to solve the problem.
# if you look at any of the popular deep learning frameworks (such as Tensorflow),
#   they're essentially graphs of linear algebra computations.
#   It's a very useful and practical way to think about machine learning algorithms.

# The neural network we're going to build for this exercise has the following layers:
#   input layer - matching the size of our instance data (400 + the bias unit).
#   hidden layer - with 25 units (26 with the bias unit).
#   output layer - with 10 units corresponding to our one-hot encoding for the class labels.

# A bias unit is an "extra" neuron added to each pre-output layer that stores the value of 1.
#   Bias units aren't connected to any previous layer and in this sense don't represent a true "activity".

#############################################

# Errors in the run:

# C:\ProgramData\Anaconda3\lib\site-packages\sklearn\preprocessing\_encoders.py:368: FutureWarning: The handling of integer data will change in version 0.22. Currently, the categories are determined based on the range [0, max(values)], while in the future they will be determined based on the unique values.
# If you want the future behaviour and silence this warning, you can specify "categories='auto'".
# In case you used a LabelEncoder before this OneHotEncoder to convert the categories to integers, then you can now use the OneHotEncoder directly.
#   warnings.warn(msg, FutureWarning)
#
# Initial cost: 6.865559809225825
#
# E:/Backup/Programming/Python/PycharmProjects/Learning/lib_scikit_learn/classification/johnwittenauer/neural_network.py:110: RuntimeWarning: divide by zero encountered in log
#   second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
# E:/Backup/Programming/Python/PycharmProjects/Learning/lib_scikit_learn/classification/johnwittenauer/neural_network.py:110: RuntimeWarning: invalid value encountered in multiply
#   second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))

#############################################

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# computes the gradient of the sigmoid function
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


# the forward-propagate function computes the hypothesis for each training instance given the current parameters
#   (in other words, given some current state of the network and a set of inputs,
#   it calculates the outputs at each layer in the network).
# The shape of the hypothesis vector (denoted by h), which contains the prediction probabilities for each class,
#   should match our one-hot encoding for y.
def forward_propagate(X, theta1, theta2):
    m = X.shape[0]

    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)

    return a1, z2, a2, z3, h


# cost function to evaluate the loss for a given set of network parameters.
# the cost function runs the forward-propagation step
#   and calculates the error of the hypothesis (predictions) vs. the true label for the instance.
# calculating the error by running the data plus current parameters through the "network"
#   (the forward-propagate function) and comparing the output to the true labels.
def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate=None):
    m = X.shape[0]
    # X = np.matrix(X)
    # y = np.matrix(y)

    # reshape \ unravel the parameter array into parameter matrices for each layer
    # theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    # theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    theta1 = np.array(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.array(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)  # computing the hypothesis matrix h

    # compute the cost
    #   applies the cost equation to compute the total error between y and h.
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m

    # the cost regularization term
    # Regularization (adds a penalty term to the cost that scales with the magnitude of the parameters)
    if learning_rate is not None:
        J += (float(learning_rate) / (2 * m)) * (
                np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    return J  # The total error across the whole dataset


# the backpropagation algorithm (cost_and_backprop).
#   backpropagation computes the parameter updates that will reduce the error of the network on the training data.
# backpropagation to compute the gradients.
# Since the computations required for backpropagation are a superset of those required in the cost function,
#   we're actually going to extend the cost function to also perform backpropagation
#   and return both the cost and the gradients.
# If you're wondering why I'm not just calling the existing cost function from within the backprop function
#   to make the design more modular, it's because backprop uses a number of other variables
#   calculated inside the cost function.
# Here's the full implementation. I skipped ahead and added gradient regularization rather than first create an un-regularized version.
def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    ##### this section is identical to the cost function logic we already saw #####
    m = X.shape[0]
    # X = np.matrix(X)
    # y = np.matrix(y)

    # reshape the parameter array into parameter matrices for each layer
    # theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    # theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    theta1 = np.array(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.array(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)

    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m

    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    ##### end of cost function logic, below is the new part #####

# this part is essentially answering the question:
#   "how can I adjust my parameters to reduce the error the next time I run through the network"?
# It does this by computing the contributions at each layer to the total error and adjusting appropriately
#   by coming up with a "gradient" matrix (or, how much to change each parameter and in what direction).

    # perform backpropagation
    for t in range(m):
        a1t = a1[t, :]  # (1, 401)
        z2t = z2[t, :]  # (1, 25)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]  # (1, 10)
        yt = y[t, :]  # (1, 10)

        d3t = ht - yt  # (1, 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)

        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    # add the gradient regularization term
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J, grad  # The total error across the whole dataset is J


#############################################

# initial setup
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1

# randomly initialize a parameter array of the size of the full network's parameters
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25

#############################################

data = loadmat('../../datasets/per_type/matlab/digits.mat')

X = data['X']
y = data['y']

# One-hot encoding our lables
#   turns a class label n (out of k classes) into a vector of length k where a single index n is "hot" (1)
#   while the rest are zero.
#   example: for classes 1-10:
#   1  = [1 0 0 0 0 0 0 0 0 0]
#   10 = [0 0 0 0 0 0 0 0 0 1]
y_onehot = OneHotEncoder(sparse=False).fit_transform(y)

# perform first backpropagation (just to get the initial cost):
J, grad = backprop(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)
print('Initial cost:', J)  # 6.865559809225825

# training our network and using it to make predictions:
# minimize the objective function:
#   We put a bound on the number of iterations since the objective function is not likely to completely converge.
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),
                method='TNC', jac=True, options={'maxiter': 250})
print(fmin)  # Final cost: 0.3873656506969604

# Using the parameters it found (fmin) and forward-propagate them through the network to get some predictions.
# this is our trained network
# We have to:
#   1. reshape the output from the optimizer (fmin) to match the parameter matrix shapes that our network is expecting,
#   2. run the forward propagation to generate a hypothesis for the input data.
# X = np.matrix(X)
# theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
# theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
theta1 = np.array(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.array(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print('accuracy = {0}%'.format(accuracy * 100))  # 97.9%

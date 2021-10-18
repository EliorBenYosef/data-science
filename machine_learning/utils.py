import numpy as np


def sigmoid_activation(z):
    """
    Sigmoid (or “logit”) activation function.

    converts a continuous input into a probability value [0,1],
        which can be interpreted as the class probability,
        or the likelihood that the input example should be classified positively.
    Using this probability along with a threshold value, we can obtain a discrete label prediction.
    """
    return 1 / (1 + np.exp(-z))


# # visualize the sigmoid activation function’s output (to see what it’s really doing):
# nums = np.arange(-10, 10, step=1)
# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(nums, sigmoid(nums), 'r')


def mse_loss(X, y, theta):
    """
    Mean Squared Error (MSE) loss.
    The cost function for linear regression.
    """
    y_pred = X * theta.T
    error = y_pred - y
    return np.sum(np.power(error, 2)) / (2 * len(X))


def log_loss(theta, x, y, learning_rate=None):
    """
    Binary cross-entropy / log loss.
    The cost function for binary (linear) logistic regression.

    Evaluates the current model's parameters' performance on the training data.
    Determines: "given some candidate solution theta applied to input X,
        how far off is the result from the true desired outcome y".

    Note that we reduce the output down to a single scalar value, which is the sum of the “error” quantified as a
    function of the difference between the class probability assigned by the model and the true label of the example.
    The implementation is completely vectorized – it’s computing the model’s predictions for the whole dataset
    in one statement (sigmoid(X * theta.T)).
    the variable called “reg” is a function of the parameter values.
        As the parameters get larger, the penalization added to the cost function increases.
    the “learning rate” parameter is also part of the regularization term in the equation.
        The learning rate gives us a new hyper-parameter that we can use to tune how much weight the regularization holds in the cost function.
    """
    # theta = np.matrix(theta)
    # x = np.matrix(X)
    # y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid_activation(x * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid_activation(x * theta.T)))
    if learning_rate is None:
        return np.sum(first - second) / (len(x))
    else:
        reg = (learning_rate / 2 * len(x)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
        return np.sum(first - second) / (len(x)) + reg


def softmax_loss(theta, x, y, learning_rate=None):
    """
    Categorical cross-entropy / softmax loss.
    The cost function for multi-class logistic regression.
    """
    # theta = np.matrix(theta)
    # x = np.matrix(X)
    # y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid_activation(x * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid_activation(x * theta.T)))
    if learning_rate is None:
        return np.sum(first - second) / (len(x))
    else:
        reg = (learning_rate / 2 * len(x)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
        return np.sum(first - second) / (len(x)) + reg


def single_gradient_step(theta, x, y, learning_rate=None):  # gradient
    """
    a function to compute the gradient of the model parameters to figure out how to change the parameters
    to improve the outcome of the model on the training data.
    Recall that with gradient descent we don’t just randomly jigger around the parameter values and see what works best.
    At each training iteration we update the parameters in a way that’s guaranteed to move them in a direction
    that reduces the training error (i.e. the “cost”).
    We can do this because the cost function is differentiable.
    Note that we don't actually perform gradient descent in this function - we just compute a single gradient step.
    the gradient function specifies how to change those parameters to get an answer that's slightly better than the one we've already got
    """
    # theta = np.matrix(theta)
    # x = np.matrix(X)
    # y = np.matrix(y)

    error = sigmoid_activation(x * theta.T) - y

    # more generalized way:
    if learning_rate is None:
        grad = ((x.T * error) / len(x)).T
    else:
        reg = (learning_rate / len(x)) * theta
        grad = ((x.T * error) / len(x)).T + reg

    grad[0, 0] = np.sum(np.multiply(error, x[:, 0])) / len(x)  # intercept gradient is not regularized
    return np.array(grad).ravel()

    # more specific way:
    # parameters = int(theta.ravel().shape[1])
    # grad = np.zeros(parameters)
    # for i in range(parameters):
    #     term = np.multiply(error, X[:, i])
    #     if learning_rate is None or i == 0:  # the first parameter is not regularized, it's considered the “bias” or “intercept” of the model and shouldn’t be penalized
    #         grad[i] = np.sum(term) / len(X)
    #     else:
    #         reg = (learningRate / len(X)) * theta[:, i]
    #         grad[i] = (np.sum(term) / len(X)) + reg
    # return grad

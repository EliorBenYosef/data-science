"""
Implementing a linear regression algorithm in python from scratch.
Methods: MSE loss minimization via Gradient Descent

https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-1/
https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-2/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from machine_learning.utils import mse_loss


multivariate = False


if multivariate:
    df = pd.read_csv('../../../datasets/per_field/sl/reg/linreg_multivariate.txt', header=None, names=['Size', 'Bedrooms', 'Price'])
else:
    df = pd.read_csv('../../../datasets/per_field/sl/reg/linreg_simple.txt', header=None, names=['Population', 'Profit'])


if multivariate:
    # Note that the scale of the values for each variable is vastly different.
    #   A house will typically have 2-5 bedrooms but may have anywhere from hundreds to thousands of square feet.
    #   If we were to run our regression algorithm on this data as-is, the "size" variable would be weighted too heavily
    #   and would end up dwarfing any contributions from the "number of bedrooms" feature.
    print(df.head(), '\n')
    # feature normalization - adjust the scale of the features to level the playing field.
    #   One way to do this is by subtracting from each value in a feature the mean of that feature,
    #   and then dividing by the standard deviation:
    df = (df - df.mean()) / df.std()  # data standardization


# Let's start by examining the data (exploratory analysis stage):
print(df.head(), '\n')
print(df.describe(), '\n')
if not multivariate:
    df.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
    plt.show()


# In order to make this cost function work seamlessly with the pandas data frame we created above,
#   we need to do some manipulating:

# 1. add ones column - append a ones column to the front of the dataset
#   we need to insert a column of 1s at the beginning of the data frame
#   in order to make the matrix operations work correctly -
#   basically it accounts for the intercept term (b) in the linear equation (y=ax+b).
df.insert(0, 'Ones', 1)

# 2. set X (training data) and y (target variable)
#   we need to separate our data into independent variables X and our dependent variable y.
cols = df.shape[1]
X = df.iloc[:, 0:cols - 1]
y = df.iloc[:, cols - 1:cols]

# 3. convert from data frames to numpy matrices, and instantiate a parameter matrix (initialize theta).
# X = np.matrix(X.values)
# y = np.matrix(y.values)
X = np.array(X.values)
y = np.array(y.values)
if multivariate:
    # theta = np.matrix(np.array([0,0,0]))  # [[0 0 0]] = ???
    theta = np.array(np.array([0,0,0]))  # [[0 0 0]] = ???
else:
    # theta = np.matrix(np.array([0,0]))  # [[b, a]] -> y = ax + b
    theta = np.array(np.array([0,0]))  # [[b, a]] -> y = ax + b

# now we can try out our cost function to get the cost (error) of the model
# Remember the parameters were initialized to 0 so the solution isn't optimal yet, but we can see if it works.
print('initial cost =', mse_loss(X, y, theta))


# a function to perform gradient descent on the parameters theta using the update rules defined in the exercise text.
def gradient_descent(X, y, theta, alpha, iters):
    # temp = np.matrix(np.zeros(theta.shape))
    temp = np.array(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y  # matrices multiplication: XR0C0*tR0C0 + XR0C1*tR1C0 = 1*b + X*a = y pred

        for j in range(parameters):  # b, a
            term = np.multiply(error, X[:, j])  # need clarification
            temp[0, j] = theta[0, j] - ((np.sum(term) / len(X)) * alpha)  # (np.sum(term) / len(X)) is the average term

        theta = temp
        cost[i] = mse_loss(X, y, theta)

    return theta, cost


# Now that we've got a way to evaluate solutions, and a way to find a good solution,
#   it's time to apply this to our dataset:

# initialize variables for learning rate and iterations
#   There is no hard and fast rule for how to initialize them and typically some trial-and-error is involved.
alpha = 0.01  # the learning rate - a factor in the update rule for the parameters that helps determine how quickly the algorithm will converge to the optimal solution.
iters = 1000  # the number of iterations.

# perform linear regression on the dataset
# perform gradient descent to "fit" the model parameters
g, cost = gradient_descent(X, y, theta, alpha, iters)
print('updated cost =', cost[0])
print('theta =', g)
print('final cost =', cost[-1])


if not multivariate:
    # Viewing The Results
    # We're now going to use matplotlib to visualize our solution.
    # overlaying a line representing our model on top of a scatter plot of the data to see how well it fits.
    # We can use numpy's "linspace" function to create an evenly-spaced series of points within the range of our data,
    # and then "evaluate" those points using our model to see what the expected profit would be.
    # We can then turn it into a line graph and plot it.

    # x = np.linspace(data.Population.min(), data.Population.max(), 100)
    x = np.linspace(0, 25, 100)

    f = g[0,0] + (g[0,1] * x)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(df.Population, df.Profit, label='Training Data')
    # ax.legend(loc=2)
    ax.legend(loc='best', shadow=False, scatterpoints=1)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.ylim(top=30, bottom=-5)
    plt.xlim(left=0, right=25)
    plt.show()


# Since the gradient decent function also outputs a vector with the cost at each training iteration,
#   plot the training progress to confirm that the error was in fact decreasing with each iteration of gradient descent.
#   we can plot that as well:

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()



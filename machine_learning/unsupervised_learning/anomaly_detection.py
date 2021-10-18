"""
https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-8/

Anomaly Detection - Supervised approach
implement an anomaly detection algorithm using a Gaussian model and apply it to detect failing servers on a network.
using a Gaussian model to detect if an unlabeled example from a dataset should be considered an anomaly.
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import stats


def estimate_gaussian(X):
    """
    Estimates a Gaussian probability distribution (by calculating the mean and variance),
    for each feature in the dataset.
    """
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)
    return mu, sigma


def select_threshold(pval, yval):
    """
    Finds the best threshold value, given the probability density values and true labels.
    Done by calculating the F1 score (a function of TP, FP, FN), for varying values of epsilon.
    """
    best_epsilon = 0
    best_f1 = 0

    step = (pval.max() - pval.min()) / 1000

    for epsilon in np.arange(pval.min(), pval.max(), step):
        preds = pval < epsilon

        tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float)
        fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)
        fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1


#########################

data = loadmat('../../datasets/per_type/matlab/ex8data1.mat')
X = data['X']  # X.shape = (307L, 2L)

# To determine a probability threshold which indicates that an example should be considered an anomaly,
#   we need to use a set of labeled validation data (where the true anomalies have been marked for us)
#   and test the model's performance at identifying those anomalies given different threshold values.
Xval = data['Xval']  # Xval.shape = (307L, 2L)
yval = data['yval']  # yval.shape = (307L, 1L)

# Getting our Gaussian model parameters (mean and variance):
mu, sigma = estimate_gaussian(X)

# Calculate the probability density of each of the values in our dataset & the validation set
#   given the Gaussian model parameters we calculated above.
# dist = stats.norm(mu, sigma)
#   calculates the normal distribution given the parameters (mean and variance)
# dist.pdf()
#   calculates the probability that the data belongs to the distribution.
#   Essentially it's computing how far each instance is from the mean
#       and how that compares to the "typical" distance from the mean for this data.
p = np.zeros((X.shape[0], X.shape[1]))  # p.shape = (307L, 2L)
p[:, 0] = stats.norm(mu[0], sigma[0]).pdf(X[:, 0])
p[:, 1] = stats.norm(mu[1], sigma[1]).pdf(X[:, 1])
pval = np.zeros((Xval.shape[0], Xval.shape[1]))
pval[:, 0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:, 0])
pval[:, 1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:, 1])

# use the validation set's probabilities combined with the true label
#   to determine the optimal probability threshold to assign data points as anomalies.
epsilon, f1 = select_threshold(pval, yval)

# apply the threshold to the dataset and visualize the results.
# indexes of the values considered to be outliers
outliers = np.where(p < epsilon)

# Visualization
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[:, 0], X[:, 1])
ax.scatter(X[outliers[0], 0], X[outliers[0], 1], s=50, color='r', marker='o')
plt.show()

# The points in red are the ones that were flagged as outliers.
# Visually these seem pretty reasonable.
# The top right point that has some separation (but was not flagged) may be an outlier too, but it's fairly close.

"""
Supervised Anomaly Detection
implement an anomaly detection algorithm using a Gaussian model and apply it to detect failing servers on a network.
using a Gaussian model to detect if an unlabeled example from a dataset should be considered an anomaly.
* using a set of labeled validation data (where the true anomalies have been marked for us)

Steps:
1. Estimate X's Probability Distribution (prob_dist).
2. Calculate the Probability Density Function at X (for each and every value in X and X_val - p, p_val).
3. Determine the optimal probability threshold value (which defines what is considered an anomaly).

https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-8/
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import stats


data = loadmat('../../../datasets/per_type/matlab/ex8data1.mat')
X = data['X']  # (307, 2)
# Using a set of labeled validation data (where the true anomalies have been marked for us):
X_val = data['Xval']  # (307, 2)
y_val = data['yval']  # (307, 1)

#########################

# 1. Estimate the Probability Distribution of X:

# Calculate the Normal (Gaussian) Probability Distribution's parameters:
#   Mean (mu) and Variance (sigma), for each feature in the dataset -
mu = X.mean(axis=0)
sigma = X.var(axis=0)

# Construct the Normal (Gaussian) Probability Distribution:
prob_dist = stats.norm(mu, sigma)

#########################

# 2. Calculate the Probability Density Function at x:
#   this is the probability that X=x - the probability that the instance (x) belongs to X's probability distribution
#       (compares its distance from the mean, to X's "typical" distance from the mean).
p = prob_dist.pdf(X)
p_val = prob_dist.pdf(X_val)

#########################

# 3. Determine the optimal probability threshold value (which defines what is considered an anomaly):


def calc_f1_score(eps, p, y):
    """
    Calculates the F1 score:
        F1 = (2 * precision * recall) / (precision + recall)
           = TP / (TP + (FP + FN) / 2)
    tests the model's performance at identifying those anomalies given different threshold values.

    :param eps: epsilon - probability threshold value
    :param p: the dataset's probabilities (that the data belongs to the distribution)
    :param y: the dataset's true labels (0/1) - in this case if it's an anomaly or not
    :return: f1 score
    """
    y_pred = p < eps  # predicts if anomaly

    tp = np.sum(np.logical_and(y_pred == 1, y == 1)).astype(float)
    fp = np.sum(np.logical_and(y_pred == 1, y == 0)).astype(float)
    fn = np.sum(np.logical_and(y_pred == 0, y == 1)).astype(float)

    precision = 0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

    return f1


def get_optimal_prob_threshold_and_score(p_val, y_val):
    """
    Finds the optimal epsilon (Probability Threshold value)
    :param p_val: probability density values
    :param y_val: true labels (anomaly / not)
    :return: best_epsilon - best probability threshold value
             best_f1 - best epsilon's f1 score
    """
    best_epsilon = 0
    best_f1 = 0

    step = (p_val.max() - p_val.min()) / 1000
    for epsilon in np.arange(p_val.min(), p_val.max(), step):  # varying values of epsilon.

        f1 = calc_f1_score(epsilon, p_val, y_val)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1


epsilon, f1 = get_optimal_prob_threshold_and_score(p_val, y_val)

#########################

# Apply the threshold to the dataset and visualize the results:
# The points in red are the ones that were flagged as outliers.
# Visually these seem pretty reasonable.
# The top right point that has some separation (but was not flagged) may be an outlier too, but it's fairly close.

# get the indexes of the values considered to be outliers:
# outliers_indices = np.where(p < epsilon)
outliers_mask = p < epsilon
outliers_indices_and = np.where(np.logical_and(outliers_mask[:, 0], outliers_mask[:, 1]))
outliers_indices_or = np.where(np.logical_or(outliers_mask[:, 0], outliers_mask[:, 1]))

# Visualization
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[:, 0], X[:, 1])
# ax.scatter(X[outliers_indices[0], 0], X[outliers_indices[0], 1], s=50, color='r', marker='o')
ax.scatter(X[outliers_indices_or, 0], X[outliers_indices_or, 1], s=50, color='r', marker='o')
plt.savefig('/results/anom_det_sup.png')
plt.show()


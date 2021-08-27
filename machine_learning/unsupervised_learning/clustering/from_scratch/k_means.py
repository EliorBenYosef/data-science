"""
https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-7/

performing clustering and image compression with K-Means and principal component analysis.
K-means clustering and principal component analysis (PCA) are both examples of unsupervised learning techniques.
Unsupervised learning problems do not have any label or target for us to learn from to make predictions,
    so unsupervised algorithms instead attempt to learn some interesting structure in the data itself.
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def init_centroids(X, k):
    """
    Randomly initializing the centroids.
    This can affect the convergence of the algorithm.
    We're tasked with creating a function that selects random examples and uses them as the initial centroids.
    """
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)

    for i in range(k):
        centroids[i, :] = X[idx[i], :]

    return centroids


def find_closest_centroids(X, centroids):
    """
    Finds the closest centroid for each instance in the data.
    """
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m, dtype=np.int)

    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i, :] - centroids[j, :]) ** 2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j

    return idx


def compute_centroids(X, idx, k):
    """
    Computes the centroid of a cluster.
    The centroid is simply the mean of all of the examples currently assigned to the cluster.
    """
    m, n = X.shape
    centroids = np.zeros((k, n))

    for i in range(k):
        indices = np.where(idx == i)
        centroids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()

    return centroids


def run_k_means(X, initial_centroids, max_iters):
    """
    K-Means Clustering implementation.
    K-means is an iterative, unsupervised clustering algorithm that groups similar instances together into clusters.
    The algorithm starts by guessing the initial centroids for each cluster, and then repeatedly assigns instances
        to the nearest cluster and re-computes the centroid of that cluster.

    running the algorithm for some number of iterations and visualizing the result.
    In order to run the algorithm we just need to alternate between assigning examples to the nearest cluster
        and re-computing the cluster centroids.
    """
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids

    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)

    return idx, centroids


######################

data = loadmat('../../../../datasets/per_type/matlab/ex7data2.mat')
X = data['X']

# initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
# idx = find_closest_centroids(X, initial_centroids)
# centroids = compute_centroids(X, idx, 3)
# print(centroids)

initial_centroids = init_centroids(X, 3)
idx, centroids = run_k_means(X, initial_centroids, 10)
idx = find_closest_centroids(X, centroids)  # get the closest centroids one last time

# We can now plot the result using color coding to indicate cluster membership.

cluster1 = X[np.where(idx == 0)[0], :]
cluster2 = X[np.where(idx == 1)[0], :]
cluster3 = X[np.where(idx == 2)[0], :]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(cluster1[:, 0], cluster1[:, 1], s=30, color='r', label='Cluster 1')
ax.scatter(cluster2[:, 0], cluster2[:, 1], s=30, color='g', label='Cluster 2')
ax.scatter(cluster3[:, 0], cluster3[:, 1], s=30, color='b', label='Cluster 3')
ax.legend()
# ax.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()


######################

# applying K-means to image compression.
# The intuition: we can use clustering to find a small number of colors that are most representative of the image,
# and map the original 24-bit colors to a lower-dimensional color space using the cluster assignments.


def compress_image(colors_num):  # colors_num = mapping the original image to only X colors
    # feeding the data into the K-means algorithm:
    initial_centroids = init_centroids(X, colors_num)
    idx, centroids = run_k_means(X, initial_centroids, 10)
    idx = find_closest_centroids(X, centroids)  # get the closest centroids one last time

    # compressing the image:
    X_recovered = centroids[idx.astype(int), :]  # map each pixel to the centroid value
    X_recovered = X_recovered.reshape(A.shape)  # reshape to the original dimensions
    return X_recovered


image_data = loadmat('../../../../datasets/per_type/matlab/bird_small.mat')  # pulling the pre-loaded raw pixel data.
A = image_data['A']  # A.shape = (128L, 128L, 3L)

# applying some preprocessing to the data:
A = A / 255.  # normalize value ranges
X = A.reshape(A.shape[0] * A.shape[1], A.shape[2])  # reshape the array

# showing the result
f, axarr = plt.subplots(2, 2)
axarr[0, 0].imshow(A)
axarr[0, 1].imshow(compress_image(32))
axarr[1, 0].imshow(compress_image(16))
axarr[1, 1].imshow(compress_image(8))
plt.savefig('../results/K-Means Image Compression.png')
plt.show()

# we created some artifacts in the compression but the main features of the image are still there
#   despite mapping the original image to less colors

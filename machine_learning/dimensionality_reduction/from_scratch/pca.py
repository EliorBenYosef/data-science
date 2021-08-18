"""
https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-7/
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def pca(X):
    """
    Principal Component Analysis (PCA) implementation.
    we'll use numpy's built-in functions to calculate the covariance and SVD of a matrix.
    :param X: input - normalized data
    :return: output - the singular value decomposition of the covariance matrix of the original data
    """
    X = (X - X.mean()) / X.std()  # normalize the features
    cov = np.dot(X.T, X) / X.shape[0]  # compute the covariance matrix
    U, S, V = np.linalg.svd(cov)  # perform SVD
    return U, S, V  # matrix U - principal components


def project_data(X, U, k):
    """
    use the principal components (matrix U) to project the original data into a lower-dimensional space.
    a function that computes the projection and selects only the top K components,
    effectively reducing the number of dimensions.
    """
    U_reduced = U[:, :k]
    return np.dot(X, U_reduced)


def recover_data(Z, U, k):
    """
    We can also attempt to recover the original data by reversing the steps we took to project it.
    """
    U_reduced = U[:, :k]
    M = np.dot(Z, U_reduced.T)
    return np.squeeze(np.asarray(M))


def apply_pca(X, n_pc):
    """
    Applying PCA
    """
    U, S, V = pca(X)
    Z = project_data(X, U, n_pc)
    # attempt to recover the original structure (and later render it again):
    X_recovered = recover_data(Z, U, n_pc)
    return X_recovered


######################

def pca_on_simple_2d_data(n_pc):
    """
    Applying PCA to a simple 2D data set.

    Notice how the points all seem to be compressed down to an invisible line.
    That invisible line is essentially the first principal component.
    The second principal component (which we cut off when we reduced the data to one dimension),
        can be thought of as the variation orthogonal to that line.
    Since we lost that information,
        our reconstruction can only place the points relative to the first principal component.
    """
    data = loadmat('../../../datasets/per_type/matlab/ex7data1.mat')
    X = data['X']

    X_recovered = apply_pca(X, n_pc)

    # visualization:
    plt.subplots(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1])
    plt.title('original data set')

    plt.subplot(1, 2, 2)
    plt.scatter(X_recovered[:, 0], X_recovered[:, 1])
    plt.title('recovered data set after PCA')

    plt.show()


def pca_for_image_compression(n_pc):
    """
    Performing image compression with PCA.

    Applying PCA to 32x32 grayscale faces images (using it to find a low-dimensional representation of the images)
    By using the same dimension reduction techniques we can capture the "essence" of the images
        using much less data than the original images.

    We lost some detail, though not as much as you might expect for a 10x reduction in the number of dimensions.
    """
    data = loadmat('../../../datasets/per_type/matlab/faces.mat')
    X = data['X']

    X_recovered = apply_pca(X, n_pc)

    # visualization:
    plt.subplots(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(np.reshape(X[3, :], (32, 32)))
    plt.title('original data set')

    plt.subplot(1, 2, 2)
    plt.imshow(np.reshape(X_recovered[3, :], (32, 32)))
    plt.title('recovered data set after PCA')

    plt.show()


pca_on_simple_2d_data(n_pc=1)
pca_for_image_compression(n_pc=100)  # take the top 100 principal components

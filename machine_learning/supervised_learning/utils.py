import numpy as np


def gaussian_kernel(x1, x2, sigma):
    """
    Implementing a gaussian kernel function.
    Although scikit-learn has a gaussian kernel built in, for transparency we'll implement one from scratch:
    """
    return np.exp(-(np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2))))

# x1 = np.array([1.0, 2.0, 1.0])
# x2 = np.array([0.0, 4.0, -1.0])
# sigma = 2
# gaussian_kernel(x1, x2, sigma)

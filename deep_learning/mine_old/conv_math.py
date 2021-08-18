import numpy as np
from scipy.ndimage import convolve


data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

kernel = np.array([[0, -1,  1],
                   [1,  0, -1],
                   [0,  1,  0]])


# padding = valid (none) \ same (output size = input size) \ full (slide starting only the bottom kernel line)
#           constant (0) \ reflect (cloning)
# stride = 1 \ 2 \ ...

# kernel transpose = [ 0,  1, 0]
#                    [-1,  0, 1]
#                    [ 1, -1, 0]

# results:

# padding valid - [3]

# padding same 0, stride 1 - [[-2, 1, -3],
#                             [-1, 3, -3],
#                             [12, 7, -2]]
# padding same 0, stride 2 - [[-2, -3],
#                             [12, -2]]

# padding same reflect, stride 1 - [[2, 3, 3],
#                                   [2, 3, 3],
#                                   [5, 6, 6]]
# padding same reflect, stride 2 - [[2, 3],
#                                   [5, 6]]

# padding full 0, stride 1 - [[0, -1, -1, -1,  3],
#                             [1, -2,  1, -3,  3],
#                             [4, -1,  3, -3,  3],
#                             [7, 12,  7, -2, -9]]
#                             [0,  7,  8,  9,  0]]


# for 1D arrays:
# convolution_product = np.convolve(data, kernel, mode='full')  # same/valid/full
# print('matrix convolution =', '\n', convolution_product, '\n')


# for matrices (2D arrays):
convolution_product = convolve(data, kernel, mode='constant')  # mode: 'reflect' (default, cloning), 'constant' (with cval, default: 0.0), 'nearest'
print('matrix convolution =', '\n', convolution_product, '\n')


# Manually (not sure if it works or efficient):
# ks = (kl-1)/2 ## kernels usually square with odd number of rows/columns
# kl = len(kernel)
# imx = len(matrix)
# imy = len(matrix[0])
# for i in range(imx):
#   for j in range(imy):
#     acc = 0
#     for ki in range(kl): ##kernel is the matrix to be used
#       for kj in range(kl):
#         if 0 <= i-ks <= kl: ## make sure you don't get out of bound error
#           acc = acc + (matrix[i-ks+ki][j-ks+kj] * kernel[ki][kj])
#   matrix[i][j] = acc




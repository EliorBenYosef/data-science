import numpy as np
import scipy.signal as sig
import math
import cv2
from kernels import scale

data = np.array([[0, 90, 0],
                 [105, 0, 55],
                 [0, 40, 0]])

# Note:
# Convolution reverses the direction of one of the functions it works on.
# one function is parameterized with τ and the other with -τ.
#   reference: https://en.wikipedia.org/wiki/Convolution#Definition
# the desired kernel must be flipped (K[::-1,::-1])) the in both axes to get the expected result.
# how it's done:
# desired_kernel = [[0, 1, 2],
#                   [3, 4, 5],
#                   [6, 7, 8]]
# used_kernel = np.array(desired_kernel)[::-1,::-1]
# used_kernel = np.flip(desired_kernel, axis=(0,1))

kernel_x = np.array([[-1, 0, 1]])[::-1, ::-1]
# kernel_x = np.flip([[-1, 0, 1]], axis=1)
kernel_y = np.array([[1], [0], [-1]])[::-1, ::-1]
# kernel_y = np.flip([[1], [0], [-1]], axis=0)

GV_x = sig.convolve2d(data, kernel_x, mode='valid')[1, 0]       # returns [[0], [-50], [0]]
GV_y = sig.convolve2d(data, kernel_y, mode='valid')[0, 1]       # returns [[0, 50, 0]]

# Gradient Vector:
GV = np.array([[GV_x], [GV_y]])
print(f'Gradient Vector: \n {GV}')

# Gradient Vector's Magnitude:
# GV_M = math.sqrt(GV_x ** 2 + GV_y ** 2)
GV_M = np.sqrt(GV_x ** 2 + GV_y ** 2)
print(f"Gradient Vector's Magnitude: {GV_M}")

# Gradient Vector's Direction (\ angle):
# GV_theta = math.degrees(math.atan(GV_y / GV_x))
# GV_theta = np.arctan(GV_y / GV_x) * 180 / np.pi
GV_theta = np.degrees(np.arctan(GV_y / GV_x))
print(f"Gradient Vector's Direction: {GV_theta}°")


##########################################

# cv2 implementation
img = cv2.imread('../../../datasets/per_field/cv/color_man_2004.jpg')
# img = np.float32(img) / 255.0  # scaling

GV_x_sobel = scale(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)).astype(np.uint8)  # cv2.CV_8U, ksize=5
GV_y_sobel = scale(cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)).astype(np.uint8)  # cv2.CV_8U, ksize=5

GV_M, GV_theta = cv2.cartToPolar(GV_x_sobel, GV_y_sobel, angleInDegrees=True)

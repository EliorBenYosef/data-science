# https://docs.scipy.org/doc/scipy/reference/

import numpy as np
from imageio import imread, imwrite

# Image operations
# SciPy provides some basic functions to work with images.
# For example, it has functions to read images from disk into numpy arrays, to write numpy arrays to disk as images, and to resize images.

img = imread('cat.jpg')  # Read an JPEG image into a numpy array. dtype = uint8; shape = (400, 248, 3)"

# tinting the image by scaling each of the color channels by a different scalar constant.
# The image has shape (400, 248, 3); we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged, and multiplies the green and blue channels by 0.95 and 0.9 respectively.
img_tinted = (img * [1, 0.95, 0.9]).astype(np.uint8)
imwrite('cat_tinted.jpg', img_tinted)  # Write the edited image back to disk

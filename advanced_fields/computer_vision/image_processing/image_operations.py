"""
Image Operations
- read images from disk into numpy arrays
- write numpy arrays to disk as images
- resize images
"""

import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from skimage import io, transform
from advanced_fields.computer_vision.image_processing.kernels import scale


input_uri = '../../../datasets/per_field/cv/color_cat.jpg'
output_uri = 'output_img/'

###################################

# imageio

# Image Reading:
img = imread(input_uri)  # Read an JPEG image into a numpy array. dtype = uint8; shape = (400, 248, 3)

# Image Tinting
#   done by scaling each of the color channels by a different scalar constant.
#   The image has shape (400, 248, 3); we multiply it by the array [1, 0.95, 0.9] of shape (3,);
#   numpy broadcasting means that this leaves the red channel unchanged, and multiplies the green and blue channels by 0.95 and 0.9 respectively.
img_tinted = (img * [1, 0.95, 0.9]).astype(np.uint8)

# Image Writing \ Saving:
#   Write the edited image back to disk
imwrite(output_uri + 'cat_tinted.jpg', img_tinted)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[0].set_title('Original')
ax[0].axis('off')
ax[1].imshow(img_tinted)
ax[1].set_title('Tinted')
ax[1].axis('off')
plt.show()

###################################

# skimage

# Image Reading:
img = scale(io.imread(input_uri))
img_grey = scale(io.imread(input_uri, as_gray=True))

# Image Resizing:
img_resized = transform.resize(img, (300, 300), mode='symmetric', preserve_range=True)
img_grey_resized = transform.resize(img_grey, (300, 300), mode='symmetric', preserve_range=True)

# Image Writing \ Saving:
io.imsave(output_uri + 'cat_resized.jpg', img_resized.astype(np.uint8))
io.imsave(output_uri + 'cat_grey.jpg', img_grey.astype(np.uint8))
io.imsave(output_uri + 'cat_grey_resized.jpg', img_grey_resized.astype(np.uint8))

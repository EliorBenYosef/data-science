"""
Histogram of Oriented Gradients (HOG)
"""

import cv2
import matplotlib.pyplot as plt

import numpy as np
import scipy.signal as sig
from image_processing_kernels import get_sobel_kernels

from skimage.feature import hog
from skimage import data, exposure

N_BUCKETS = 9
CELL_SIZE = 8  # Each cell is CELL_SIZExCELL_SIZE pixels
BLOCK_SIZE = 2  # Each block is BLOCK_SIZExBLOCK_SIZE cells

##########################################

# scipy implementation

# N_BUCKETS = 9
# CELL_SIZE = 8
# BLOCK_SIZE = 2

dd = 180 / N_BUCKETS  # delta degree between bins


def get_magnitude_values_per_direction_bucket(m, d):
    """
    :param m: flattened magnitudes
    :param d: flattened directions
    """
    magnitude_values = np.zeros(N_BUCKETS)  # per direction bucket

    left_bin = (d // dd).astype(int)
    right_bin = left_bin + 1

    left_val = m * (right_bin * dd - d) / dd
    right_val = m * (d - left_bin * dd) / dd

    magnitude_values[left_bin] += left_val
    magnitude_values[right_bin] += right_val

    return magnitude_values


def get_magnitudes_and_direction(r, c):
    """
    (r, c) defines the top left corner of the target cell.
    """
    cell_x = GV_x[r:r + CELL_SIZE, c:c + CELL_SIZE]
    cell_y = GV_y[r:r + CELL_SIZE, c:c + CELL_SIZE]
    magnitudes = np.sqrt(cell_x ** 2 + cell_y ** 2)
    for i in range(CELL_SIZE):
        for j in range(CELL_SIZE):
            if cell_y[i, j] == 0 and cell_x[i, j] == 0:  # y/x will produce nan value
                cell_x[i, j] = 1  # the value doesn't matter as 0/x=0
    directions = np.abs(np.degrees(np.arctan(cell_y / cell_x)))
    directions[directions == 180] = 0  # assigns degree 180 to 0 (bin 9 to 0)
    return magnitudes, directions


def plot_hog_single(magnitude_hist, bucket_names, title='cell'):
    plt.figure(figsize=(10, 3))
    plt.bar(range(len(magnitude_hist)), magnitude_hist / np.linalg.norm(magnitude_hist), align='center', alpha=0.8,
            width=0.9)
    plt.xticks(range(len(magnitude_hist)), bucket_names * dd, rotation=90)
    plt.xlabel('Direction buckets')
    plt.ylabel('Magnitude')
    plt.grid(ls='--', color='k', alpha=0.1)
    plt.title("HOG of %s [%d, %d]" % (title, r * CELL_SIZE, c * CELL_SIZE))
    plt.tight_layout()
    plt.show()


img = cv2.imread('manu-2004.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (img.shape[1] - img.shape[1] % CELL_SIZE, img.shape[0] - img.shape[0] % CELL_SIZE))
# img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

K_sobel_x, K_sobel_y = get_sobel_kernels()
GV_x = sig.convolve2d(img, K_sobel_x[::-1, ::-1], mode='same')
GV_y = sig.convolve2d(img, K_sobel_y[::-1, ::-1], mode='same')

cell_rows = img.shape[0] // CELL_SIZE
cell_columns = img.shape[1] // CELL_SIZE
magnitude_hists = np.zeros((cell_rows, cell_columns, N_BUCKETS))

for r in range(cell_rows):
    for c in range(cell_columns):
        # for each cell:
        magnitudes, directions = get_magnitudes_and_direction(r * CELL_SIZE, c * CELL_SIZE)
        magnitude_hist = get_magnitude_values_per_direction_bucket(magnitudes.flatten(), directions.flatten())
        magnitude_hists[r, c] = magnitude_hist
        plot_hog_single(magnitude_hist, bucket_names=np.arange(N_BUCKETS))

for r in range(cell_rows - BLOCK_SIZE + 1):
    for c in range(cell_columns - BLOCK_SIZE + 1):
        # for each block:
        magnitude_hists_block = np.resize(magnitude_hists[r:r + BLOCK_SIZE, c:c + BLOCK_SIZE],
                                          (N_BUCKETS * (BLOCK_SIZE ** 2)))
        plot_hog_single(magnitude_hists_block, bucket_names=np.tile(np.arange(N_BUCKETS), BLOCK_SIZE ** 2),
                        title='block')


##########################################

# skimage implementation

# N_BUCKETS = 8
# CELL_SIZE = 16
# BLOCK_SIZE = 1

def plot_hog_image(img, hog_img):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(img, cmap='gray')  # cmap=plt.cm.gray
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap='gray')  # cmap=plt.cm.gray
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()


img = data.astronaut()
# img = cv2.imread('manu-2004.jpg')

fd, hog_img = hog(img, orientations=N_BUCKETS, pixels_per_cell=(CELL_SIZE, CELL_SIZE),
                  cells_per_block=(BLOCK_SIZE, BLOCK_SIZE), visualize=True, multichannel=True)

plot_hog_image(img, hog_img)

##########################################

# cv2 implementation
# https://github.com/opencv/opencv/blob/master/samples/python/hist.py
# https://github.com/opencv/opencv/blob/master/samples/python/color_histogram.py

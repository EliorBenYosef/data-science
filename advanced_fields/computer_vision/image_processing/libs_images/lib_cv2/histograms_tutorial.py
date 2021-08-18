# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_begins/py_histogram_begins.html

from __future__ import print_function
from __future__ import division
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Histogram - finding, plotting, analyzing

# cv2.calcHist(images, channels, mask, histSize, ranges, hist, accumulate):
#   images : the source image of type uint8 or float32. should be given in square brackets, ie, “[img]”.
#   channels : the index of channel for which we calculate histogram. also given in square brackets.
#       For grayscale image, pass [0].
#       For color image, you can pass [0],[1],[2] for BGR channel respectively.
#   mask : mask image.
#       To find histogram of full image - pass “None”.
#       To find histogram of particular region of image, you have to create a mask image for that and give it as mask.
#   histSize : the BIN count. given in square brackets. For full scale, we pass [256].
#   ranges : possible pixel values range. Normally, it is [0,256].

histSize = 256  # number of bins
histPixelValuesRange = [0, 256]  # can also be a tuple. the upper boundary is exclusive.

##################################################

# Grayscale Histogram

def show_grayscale_histogram(img):

    # [Compute the histogram]
    hist = cv2.calcHist([img], [0], None, [histSize], histPixelValuesRange)

    # [Draw the histograms]
    hist_w = 512
    hist_h = 400
    bin_w = int(round(hist_w / histSize))
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

    # [Normalize the result to ( 0, histImage.rows )]
    cv2.normalize(hist, hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)

    # [Draw for each channel]
    for i in range(1, histSize):
        cv2.line(histImage,
                 (bin_w * (i - 1), hist_h - int(round(hist[i - 1][0]))),
                 (bin_w * i, hist_h - int(round(hist[i][0]))),
                 (255, 255, 255), thickness=2)

    # [Display]
    cv2.imshow('calcHist Demo', histImage)


img = cv2.imread('lady.jpg', 0)
cv2.imshow('Source image', img)
show_grayscale_histogram(img)
cv2.waitKey()


##################################################

# RGB Histogram
# https://github.com/opencv/opencv/blob/master/samples/python/tutorial_code/Histograms_Matching/histogram_calculation/calcHist_Demo.py

img = cv2.imread('lady.jpg')
bgr_planes = cv2.split(img)  # [Separate the image in 3 places ( B, G and R )]

# [Compute the histograms]
b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histPixelValuesRange)
g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histPixelValuesRange)
r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histPixelValuesRange)

# [Draw the histograms]
hist_w = 512
hist_h = 400
bin_w = int(round(hist_w / histSize))
histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

# [Normalize the result to ( 0, histImage.rows )]
cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)

# [Draw for each channel]
for i in range(1, histSize):
    cv2.line(histImage,
             (bin_w *(i-1), hist_h - int(round(b_hist[i-1][0]))),
             (bin_w * i, hist_h - int(round(b_hist[i][0]))),
             (255, 0, 0), thickness=2)
    cv2.line(histImage,
             (bin_w *(i-1), hist_h - int(round(g_hist[i-1][0]))),
             (bin_w * i, hist_h - int(round(g_hist[i][0]))),
             (0, 255, 0), thickness=2)
    cv2.line(histImage,
             (bin_w *(i-1), hist_h - int(round(r_hist[i-1][0]))),
             (bin_w * i, hist_h - int(round(r_hist[i][0]))),
             (0, 0, 255), thickness=2)

# [Display]
cv2.imshow('Source image', img)
cv2.imshow('calcHist Demo', histImage)
cv2.waitKey()


##################################################

# Grayscale Histogram - using Mask

img = cv2.imread('lady.jpg', 0)

# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv2.bitwise_and(img, img, mask=mask)

# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv2.calcHist([img], [0], None, [histSize], histPixelValuesRange)
hist_mask = cv2.calcHist([img], [0], mask, [histSize], histPixelValuesRange)

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])

plt.show()

from __future__ import print_function
from __future__ import division
import cv2
import numpy as np


# https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html

def show_grayscale_histogram(img, histSize):  # histSize - number of bins
    # [Compute the histogram]
    hist = cv2.calcHist([img], [0], None, [histSize], [0, 256])

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


img = cv2.imread('../../../../datasets/per_type/img/color_image.jpg', 0)
cv2.imshow('Source image', img)
cv2.waitKey()
show_grayscale_histogram(img, 256)
cv2.waitKey()
show_grayscale_histogram(img, 16)
cv2.waitKey()
show_grayscale_histogram(img, 2)
cv2.waitKey()

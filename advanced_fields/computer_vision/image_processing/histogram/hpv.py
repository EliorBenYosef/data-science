"""
Histograms of Pixel Values

https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_begins/py_histogram_begins.html
https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html

Histogram - finding, plotting, analyzing

cv2.calcHist(images, channels, mask, histSize, ranges, hist, accumulate):
  images : the source image of type uint8 or float32. should be given in square brackets, ie, “[img]”.
  channels : the index of channel for which we calculate histogram. also given in square brackets.
      For grayscale image, pass [0].
      For color image, you can pass [0],[1],[2] for BGR channel respectively.
  mask : mask image.
      To find histogram of full image - pass “None”.
      To find histogram of particular region of image, you have to create a mask image for that and give it as mask.
  histSize : the BIN count. given in square brackets. For full scale, we pass [256].
  ranges : possible pixel values range. Normally, it is [0,256].
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import data


def get_hist_gray(img_gray, n_bins=256, pixel_range=(0, 256)):
    """
    Computes and returns Grayscale Histogram
    :param img_gray:
    :param n_bins: number of bins
    :param pixel_range: the pixel values range of the histogram (the upper boundary is exclusive)
    :return:
    """
    # Compute the histogram:
    hist = cv2.calcHist(images=[img_gray], channels=[0], mask=None, histSize=[n_bins], ranges=pixel_range)

    # # using Mask:
    # mask = np.zeros(img.shape[:2], np.uint8)
    # mask[100:300, 100:400] = 255
    # # masked_img = cv2.bitwise_and(img, img, mask=mask)
    # hist_mask = cv2.calcHist(images=[img], channels=[0], mask=mask, histSize=[n_bins], ranges=pixel_range)

    # hist_img = draw_hist([hist], [(255, 255, 255)], n_bins)
    return hist


def get_rgb_histogram(img_bgr, n_bins=256, pixel_range=(0, 256)):
    """
    Computes and returns RGB Histogram
    https://github.com/opencv/opencv/blob/master/samples/python/tutorial_code/Histograms_Matching/histogram_calculation/calcHist_Demo.py
    :param img_bgr:
    :param n_bins: number of bins
    :param pixel_range: the pixel values range of the histogram (the upper boundary is exclusive)
    :return:
    """
    bgr_planes = cv2.split(img_bgr)  # [Separate the image in 3 places ( B, G and R )]
    # Compute the histograms:
    hist_b = cv2.calcHist(images=bgr_planes, channels=[0], mask=None, histSize=[n_bins], ranges=pixel_range)
    hist_g = cv2.calcHist(images=bgr_planes, channels=[1], mask=None, histSize=[n_bins], ranges=pixel_range)
    hist_r = cv2.calcHist(images=bgr_planes, channels=[2], mask=None, histSize=[n_bins], ranges=pixel_range)
    hists = [hist_b, hist_g, hist_r]

    # colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    # hist_img = draw_hist(hists, colors, n_bins)
    return hists


def draw_hist_cv2(hists, colors, n_bins=256, w=512, h=400):
    """
    Draw the histogram/s
    """
    bin_w = int(round(w / n_bins))
    hist_img = np.zeros((h, w, 3), dtype=np.uint8)

    # [Normalize the result to ( 0, hist_img.rows )]
    for hist in hists:
        cv2.normalize(hist, hist, alpha=0, beta=h, norm_type=cv2.NORM_MINMAX)

    # [Draw for each channel]
    for i in range(1, n_bins):
        for hist, color in zip(hists, colors):
            cv2.line(hist_img,
                     (bin_w * (i - 1), h - int(round(hist[i - 1][0]))),
                     (bin_w * i, h - int(round(hist[i][0]))),
                     color, thickness=2)

    return hist_img


if __name__ == '__main__':
    # img_gray = cv2.imread('../../../../datasets/per_field/cv/color_lady.jpg', cv2.IMREAD_GRAYSCALE)
    # img_bgr = cv2.imread('../../../../datasets/per_field/cv/color_lady.jpg')

    img = data.astronaut()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    hist_gray = get_hist_gray(img_gray)
    hists_bgr = get_rgb_histogram(img_bgr)

    plt.subplot(221), plt.axis('off'), plt.imshow(img_gray, 'gray')

    ax2 = plt.subplot(222)
    plt.xlim([0, 256])
    plt.xticks(range(0, 256, 50))
    plt.plot(hist_gray, color='k')
    ax2.set_title('Grayscale Histogram')

    plt.subplot(223), plt.axis('off'), plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), 'gray')

    ax4 = plt.subplot(224)
    plt.xlim([0, 256])
    plt.xticks(range(0, 256, 50))
    plt.plot(hists_bgr[0], color='b')
    plt.plot(hists_bgr[1], color='g')
    plt.plot(hists_bgr[2], color='r')
    ax4.set_title('RGB Histogram')

    plt.tight_layout()
    plt.savefig('results/hpv.png')
    plt.show()

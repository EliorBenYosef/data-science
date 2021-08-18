import cv2
import numpy as np
import random as rnd
from matplotlib import pyplot as plt


def rand_float_range(base):
    margin = 0.03
    return base - margin + rnd.random() * margin * 2


img = cv2.imread('../../../../datasets/per_type/img/color_image.jpg')
rows, cols = img.shape[:2]

M = np.float32([[rand_float_range(1), rand_float_range(0), rnd.randrange(-cols, cols) / 4],
                [rand_float_range(0), rand_float_range(1), rnd.randrange(-rows, rows) / 4]])

dst = cv2.warpAffine(img, M, (cols, rows))

# cv2.imshow('Translation',dst)
# cv2.waitKey()

plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()

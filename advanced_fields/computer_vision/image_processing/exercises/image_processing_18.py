# https://docs.microsoft.com/en-us/dotnet/framework/winforms/advanced/why-transformation-order-is-significant

import cv2
import numpy as np
from matplotlib import pyplot as plt


def translate(img):
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    return cv2.warpAffine(img, M, (cols, rows))


def scale(img):
    return cv2.resize(img, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)


def rotate(img):
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    return cv2.warpAffine(img, M, (cols, rows))


# 1. translate --> scale VS. scale --> translate
# בגלל שהערך שניתן לפונקציה הוא מספר קבוע, יש לו משמעות שונה ביחס לגודל התמונה הסופי לאחר תרגום
img = cv2.imread('../../../../datasets/per_type/img/color_image.jpg')
plt.subplot(121), plt.imshow(scale(translate(img))), plt.title('translate --> scale')
plt.subplot(122), plt.imshow(translate(scale(img))), plt.title('scale --> translate')
plt.show()

# 2. translate --> rotate VS. rotate --> translate
#
img = cv2.imread('../../../../datasets/per_type/img/color_image.jpg')
plt.subplot(121), plt.imshow(rotate(translate(img))), plt.title('translate --> rotate')
plt.subplot(122), plt.imshow(translate(rotate(img))), plt.title('rotate --> translate')
plt.show()

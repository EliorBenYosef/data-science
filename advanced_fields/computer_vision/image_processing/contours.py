import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

dir = '../../../datasets/per_field/cv/'
file = 'color_lady.jpg'

# img_pil = Image.open(path)
# img_rgb = np.array(img_pil.convert('RGB'))  # if the img has an additional alpha channel
# img_bgr = img_rgb.copy()[:, :, ::-1]  # convert to BGR. last dimension is RGB --> BGR
# img_gray = np.array(img_pil.convert('L'))

img = cv2.imread(dir + file)  # note that OpenCV uses BGR (and not RGB)
img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # if the img has an additional alpha channel
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

thresh, img_bw = cv2.threshold(img_gray, thresh=128, maxval=255, type=0)

im2, contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# im2, contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# im2, contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

img_cont = img_bgr.copy()  # since OpenCV expects BGR (and not RGB)
cv2.drawContours(img_cont, contours, -1, (0, 255, 0), 3)  # cv2.FILLED instead of 3 fills objects

cv2.imwrite('img_' + file, img)
cv2.imwrite('bgr_' + file, img_bgr)
cv2.imwrite('rgb_' + file, img_rgb)
cv2.imwrite('bw_' + file, img_bw)
cv2.imwrite('cnt_' + file, img_cont)

# plt.imshow(img_cont)
# plt.show()

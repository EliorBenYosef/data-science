import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

dir = '../../../datasets/per_field/cv/'
file_name = 'color_lady'
file_extension = '.jpg'
path = dir + file_name + file_extension

# img_pil = Image.open(path)
# img_rgb = np.array(img_pil.convert('RGB'))  # if the img has an additional alpha channel
# img_bgr = img_rgb.copy()[:, :, ::-1]  # convert to BGR. last dimension is RGB --> BGR
# img_gray = np.array(img_pil.convert('L'))

img_bgr = cv2.imread(path)  # note that OpenCV uses BGR (and not RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
thresh, img_bw = cv2.threshold(img_gray, thresh=128, maxval=255, type=0)

contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

img_bw_cont = cv2.cvtColor(img_bw, cv2.COLOR_GRAY2BGR)
img_gray_cont = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
img_bgr_cont = img_bgr.copy()  # since OpenCV expects BGR (and not RGB)

cv2.drawContours(img_bw_cont, contours, -1, (0, 255, 0), 3)  # cv2.FILLED instead of 3 fills objects
cv2.drawContours(img_gray_cont, contours, -1, (0, 255, 0), 3)  # cv2.FILLED instead of 3 fills objects
cv2.drawContours(img_bgr_cont, contours, -1, (0, 255, 0), 3)  # cv2.FILLED instead of 3 fills objects

cv2.imwrite('output_img/' + file_name + '_01' + file_extension, img_bgr)  # '_bgr'
cv2.imwrite('output_img/' + file_name + '_02' + file_extension, img_gray)  # '_gray'
cv2.imwrite('output_img/' + file_name + '_03' + file_extension, img_bw)  # '_bw'
cv2.imwrite('output_img/' + file_name + '_04' + file_extension, img_bw_cont)  # '_bw_cont'
cv2.imwrite('output_img/' + file_name + '_05' + file_extension, img_gray_cont)  # '_gray_cont'
cv2.imwrite('output_img/' + file_name + '_06' + file_extension, img_bgr_cont)  # '_bgr_cont'

# plt.imshow(img_bgr_cont)
# plt.show()

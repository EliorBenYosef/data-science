import cv2

dir = '../../../../datasets/per_field/cv/'
file_name = 'color_lady'
file_extension = '.jpg'
path = dir + file_name + file_extension

img_bgr = cv2.imread(path)  # note that OpenCV uses BGR (and not RGB)
# img_bgra = cv2.imread(path)  # note that OpenCV uses BGR (and not RGB)
# img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)  # if the img has an additional alpha channel
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
# img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

cv2.imwrite(file_name + '_gray' + file_extension, img_gray)
cv2.imwrite(file_name + '_bgr' + file_extension, img_bgr)
cv2.imwrite(file_name + '_rgb' + file_extension, img_rgb)

cv2.imshow('Source Image (Gray)', img_gray)
cv2.imshow('Source Image (BGR)', img_bgr)
cv2.imshow('Source Image (RGB)', img_rgb)
cv2.waitKey()

# plt.imshow(img_gray)
# plt.imshow(img_bgr)
# plt.imshow(img_rgb)

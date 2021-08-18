import cv2
import numpy as np

# create an image: black background, white filled rectangle, additive noise std=3

img = np.zeros([500, 500, 3], dtype=np.uint8)

bottom_left_corner = (200, 300)
top_right_corner = (400, 200)
color = (255, 255, 255)
img_with_rect = cv2.rectangle(img, bottom_left_corner, top_right_corner, color, cv2.FILLED)

mean = 0.0  # mean
std = 3.0  # standard deviation
noisy_img = img_with_rect + np.random.normal(mean, std, img_with_rect.shape)
# noisy_img_clipped = np.clip(noisy_img, 0, 255)  # we might get out of bounds due to noise

cv2.imwrite('../../../../datasets/per_type/img/Q20_img.jpg', noisy_img)
cv2.imshow("my image", img)
cv2.waitKey()

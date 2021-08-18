import cv2
import numpy as np

image_pixel_size = 500
color_cube_size = 100

color_1 = np.random.randint(0, 256, (3,))
color_2 = np.random.randint(0, 256, (3,))

img = np.zeros([image_pixel_size, image_pixel_size, 3], dtype=np.uint8)
for row in range(image_pixel_size):
    for column in range(image_pixel_size):
        if (row // color_cube_size) % 2 == (column // color_cube_size) % 2:
            red, green, blue = color_1
        else:
            red, green, blue = color_2

        img[row, column, 0] = blue
        img[row, column, 1] = green
        img[row, column, 2] = red

cv2.imwrite('../../../../datasets/per_type/img/generated_chess_board.jpg', img)
cv2.imshow("my chess board", img)
cv2.waitKey()

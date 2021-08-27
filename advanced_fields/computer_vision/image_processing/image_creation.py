# mode='L' forces the image to be parsed in the grayscale.

import numpy as np
import cv2
from PIL import Image

h = 500
w = 500

img = np.ones([h, w, 3], dtype=np.uint8)
img = img * np.random.randint(0, 256, (3,), dtype=np.uint8)
# img[:, :, 0] = img[:, :, 0] * np.random.randint(0, 256)
# img[:, :, 1] = img[:, :, 1] * np.random.randint(0, 256)
# img[:, :, 2] = img[:, :, 2] * np.random.randint(0, 256)

cv2.imwrite('output_img/generated_image.jpg', img)
cv2.imshow('my image', img)
cv2.waitKey()

########################################

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

cv2.imwrite('output_img/generated_chess_board.jpg', img)
cv2.imshow("my chess board", img)
cv2.waitKey()

########################################

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

cv2.imwrite('../output_img/Q20_img.jpg', noisy_img)
cv2.imshow("my image", img)
cv2.waitKey()

########################################


def show_white_image_with_vertical_lines(w, h):
    img_pil = Image.new('L', (w, h), 'white')
    pixel_matrix = img_pil.load()

    for x in range(w):
        if x % 10 == 0:
            for y in range(h):
                pixel_matrix[x, y] = 0

    img_pil.show()


def show_white_image_with_diagonal_line(n):
    img_pil = Image.new('L', (n, n), 'white')
    pixel_matrix = img_pil.load()

    # less efficient
    # for x in range(n):
    #     for y in range(n):
    #         if x == y:
    #             pixel_matrix[x,y] = 0

    # more efficient
    for x in range(n):
        pixel_matrix[x, x] = 0

    img_pil.show()


def show_white_image_with_circles(n):
    img_pil = Image.new(mode='L', size=(n, n))
    pixel_matrix = img_pil.load()

    for x in range(n):
        for y in range(n):
            pixel_matrix[x, y] = round((x - (n // 2)) ** 2 + (y - (n // 2)) ** 2) % (n // 2)

    img_pil.show()


def show_psychedelic_image(n):
    img_pil = Image.new('L', (n, n), 'white')
    pixel_matrix = img_pil.load()

    for x in range(n):
        for y in range(n):
            pixel_matrix[x, y] = ((x - (n // 2)) * (y - (n // 2))) % (n // 2)

    img_pil.show()


def show_psychedelic_image_mine(n):
    img_pil = Image.new('L', (n, n), 'white')
    pixel_matrix = img_pil.load()

    for x in range(n):
        for y in range(n):
            if y % 2 == 0:
                if x % 2 == 0:
                    pixel_matrix[x, y] = ((x + 100) // 2 + (y - 100) ** 2) % (n // 2)
                else:
                    pixel_matrix[x, y] = ((x + 100) ** 2 + (y + 100) // 2) % (n // 2)
            else:
                if x % 2 == 0:
                    pixel_matrix[x, y] = round((x - (n // 2)) ** 2 + (y - (n // 2)) ** 2) % (n // 2)
                else:
                    pixel_matrix[x, y] = round((x - (n // 2)) // 2 + (y - (n // 2)) // 2) % (n // 2)

    img_pil.show()


def show_image_01(n):
    img_pil = Image.new('L', (n, n), 255)
    pixel_matrix = img_pil.load()

    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0:
                pixel_matrix[i, j] = 0

    img_pil.show()


def show_image_02(n):
    img_pil = Image.new('L', (n, n), 255)
    pixel_matrix = img_pil.load()

    for i in range(n):
        for j in range(n):
            if i <= j:
                pixel_matrix[i, j] = 0

    img_pil.show()


def show_image_03(n):
    img_pil = Image.new('L', (n, n), 255)
    pixel_matrix = img_pil.load()

    for i in range(n):
        for j in range(n):
            pixel_matrix[i, j] = 20 * i

    img_pil.show()


# show_white_image_with_vertical_lines(50, 100)
# show_white_image_with_diagonal_line(500)
# show_white_image_with_circles(512)
# show_psychedelic_image(512)
# show_psychedelic_image_mine(512)
# show_image_01(500)
# show_image_02(500)
# show_image_03(500)

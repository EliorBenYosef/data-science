import cv2
import numpy as np
import random as rnd
import glob

img = cv2.imread('../../../../datasets/per_field/cv/color_lady.jpg', cv2.IMREAD_COLOR)
rows, cols = img.shape[:2]

# Translation (3rd option):
M = np.float32([[1, 0, rnd.randrange(-cols, cols)], [0, 1, rnd.randrange(-rows, rows)]])
# M = np.float32([[rnd.random(),0,rnd.randrange(-cols,cols)],[0,rnd.random(),rnd.randrange(-rows,rows)]])
# M = np.float32([[rnd.random(),rnd.random(),rnd.randrange(-cols,cols)],[rnd.random(),rnd.random(),rnd.randrange(-rows,rows)]])

img_translate = cv2.warpAffine(img, M, (cols, rows))

###############################

img_array = img_translate
size = (100, 100)

out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

###############################

img_array = []
for filename in glob.glob('images/*.jpg'):
    img = cv2.imread(filename)
    # height, width, layers = img.shape
    # size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

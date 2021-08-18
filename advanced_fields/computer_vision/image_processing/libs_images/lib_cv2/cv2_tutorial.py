
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_basic_ops/py_basic_ops.html

# https://github.com/opencv/opencv/tree/master/samples/python


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = np.zeros([500,500,3], dtype=np.uint8)
img[:,:,0] = np.ones([500,500])*64/255.0
img[:,:,1] = np.ones([500,500])*128/255.0
img[:,:,2] = np.ones([500,500])*192/255.0

cv2.imwrite('generated_image.jpg', img)
cv2.imshow("my image", img)
cv2.waitKey()

otherImage = cv2.imread('lady.jpg', cv2.IMREAD_COLOR)


###############################################

# Geometric Transformations of Images
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html

# there are two main functions here:
#   cv2.warpAffine()
#   cv2.warpPerspective()

img = cv2.imread('lady.jpg', cv2.IMREAD_COLOR)
height,width = img.shape[:2]
rows,cols = img.shape[:2]
# rows,cols,ch = img.shape

# Scaling - resizing of the image
#   The size of the image can be specified manually, or you can specify the scaling factor.
#   Different interpolation methods are used.
#   Preferable interpolation methods are:
#       cv2.INTER_AREA for shrinking
#       cv2.INTER_CUBIC (slow) & cv2.INTER_LINEAR (default) for zooming
# 1st option:
res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
cv2.imshow('Scaling', res)
cv2.waitKey(0)
# 2nd option:
res = cv2.resize(img, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)
cv2.imshow('Scaling', res)
cv2.waitKey(0)

# Translation - shifting the object’s location
#   If you know the shift in (x,y) direction, let it be (t_x,t_y), you can create the 2x3 transformation matrix M as follows:
#   M = [[1,0,t_x],[0,1,t_y]] = 2x2 unit\identity matrix + translation vector
#   You can take make it into a Numpy array of type np.float32 and pass it into cv2.warpAffine() function.
# Warning - Third argument of the cv2.warpAffine() function is the size of the output image,
#   which should be in the form of (width, height) = (number of columns, number of rows).
M = np.float32([[1,0,100],[0,1,50]])
dst = cv2.warpAffine(img, M, (cols, rows))
cv2.imshow('Translation', dst)
cv2.waitKey()
# cv2.destroyAllWindows()

# Rotation:
#   OpenCV provides scaled rotation with adjustable rotation center so that you can rotate at any location you prefer.
#   To find this modified transformation matrix, OpenCV provides a function, cv2.getRotationMatrix2D.
# Check below example which rotates the image by 90 degree with respect to center without any scaling:
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
dst = cv2.warpAffine(img, M, (cols, rows))
cv2.imshow('Rotation', dst)
cv2.waitKey()
# cv2.destroyAllWindows()

# Affine Transformation
#   all parallel lines in the original image will still be parallel in the output image.
#   To find the transformation matrix, we need 3 points from input image and their corresponding locations in output image.
#   Then cv2.getAffineTransform will create a 2x3 matrix which is to be passed to cv2.warpAffine.
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.title('Affine Transformation')
plt.show()

# Perspective Transformation
#   For this you need a 3x3 transformation matrix.
#   Straight lines will remain straight even after the transformation.
#   To find this transformation matrix, you need 4 points on the input image and corresponding points on the output image.
#   Among these 4 points, 3 of them should not be collinear.
#   Then transformation matrix can be found by the function cv2.getPerspectiveTransform.
#   Then apply cv2.warpPerspective with this 3x3 transformation matrix.
rows,cols,ch = img.shape
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (300, 300))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.title('Perspective Transformation')
plt.show()


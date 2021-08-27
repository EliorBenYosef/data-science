"""
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_basic_ops/py_basic_ops.html

https://github.com/opencv/opencv/tree/master/samples/python

https://docs.microsoft.com/en-us/dotnet/framework/winforms/advanced/why-transformation-order-is-significant
since the function's input value is a scalar (a fixed number),
it has a different meaning in relation to the final image size after translation
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import random as rnd


# Geometric Transformations of Images
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html

# there are two main functions here:
#   cv2.warpAffine()
#   cv2.warpPerspective()

img = cv2.imread('../../../datasets/per_field/cv/color_lady.jpg', cv2.IMREAD_COLOR)
# height, width = img.shape[:2]
rows, cols = img.shape[:2]
# rows, cols, ch = img.shape


def scale(img, h_ratio=0.8, w_ratio=0.8):
    """
    Scaling - resizing of the image
        The size of the image can be specified manually, or you can specify the scaling factor.
        Different interpolation methods are used.
        Preferable interpolation methods are:
            cv2.INTER_AREA for shrinking
            cv2.INTER_CUBIC (slow) & cv2.INTER_LINEAR (default) for zooming
    """
    return cv2.resize(img, None, fx=w_ratio, fy=h_ratio, interpolation=cv2.INTER_CUBIC)  # 1st option
    # return cv2.resize(img, (w_ratio * width, h_ratio * height), interpolation=cv2.INTER_CUBIC)  # 2nd option


def translate(img):
    """
    Translation - shifting the object’s location
        If you know the shift in (x,y) direction, let it be (t_x,t_y), you can create the 2x3 transformation matrix M as follows:
        M = [[1,0,t_x],[0,1,t_y]] = 2x2 unit\identity matrix + translation vector
        You can take make it into a Numpy array of type np.float32 and pass it into cv2.warpAffine() function.
    Warning - Third argument of the cv2.warpAffine() function is the size of the output image,
        which should be in the form of (width, height) = (number of columns, number of rows).
    """
    # 1st option:
    M = np.float32([[1, 0, 100], [0, 1, 50]])

    # # 2nd option:
    # def rand_float_range(base):
    #     margin = 0.03
    #     return base - margin + rnd.random() * margin * 2
    # M = np.float32([[rand_float_range(1), rand_float_range(0), rnd.randrange(-cols, cols) / 4],
    #                 [rand_float_range(0), rand_float_range(1), rnd.randrange(-rows, rows) / 4]])

    # # 3rd option:
    # M = np.float32([[1, 0, rnd.randrange(-cols, cols)], [0, 1, rnd.randrange(-rows, rows)]])
    # # M = np.float32([[rnd.random(),0,rnd.randrange(-cols,cols)],[0,rnd.random(),rnd.randrange(-rows,rows)]])
    # # M = np.float32([[rnd.random(),rnd.random(),rnd.randrange(-cols,cols)],[rnd.random(),rnd.random(),rnd.randrange(-rows,rows)]])

    return cv2.warpAffine(img, M, (cols, rows))


def rotate(img):
    """
    Rotation:
        OpenCV provides scaled rotation with adjustable rotation center so that you can rotate at any location you prefer.
        To find this modified transformation matrix, OpenCV provides a function, cv2.getRotationMatrix2D.
    Check below example which rotates the image by 90 degree with respect to center without any scaling:
    """
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    return cv2.warpAffine(img, M, (cols, rows))


def affine_transform(img):
    """
    Affine Transformation
        all parallel lines in the original image will still be parallel in the output image.
        To find the transformation matrix, we need 3 points from input image and their corresponding locations in output image.
        Then cv2.getAffineTransform will create a 2x3 matrix which is to be passed to cv2.warpAffine.
    """
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (cols, rows))


def perspective_transform(img):
    """
    Perspective Transformation
        For this you need a 3x3 transformation matrix.
        Straight lines will remain straight even after the transformation.
        To find this transformation matrix, you need 4 points on the input image and corresponding points on the output image.
        Among these 4 points, 3 of them should not be collinear.
        Then transformation matrix can be found by the function cv2.getPerspectiveTransform.
        Then apply cv2.warpPerspective with this 3x3 transformation matrix.
    """
    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (300, 300))


if __name__ == '__main__':
    cv2.imshow('Scaling', scale(img))
    cv2.imshow('Translation', translate(img))
    cv2.imshow('Rotation', rotate(img))
    cv2.imshow('Affine Transformation', affine_transform(img))
    cv2.imshow('Perspective Transformation', perspective_transform(img))
    cv2.waitKey()
    # cv2.destroyAllWindows()

    ##############################

    def bgr2rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(bgr2rgb(img))
    ax[0, 1].imshow(bgr2rgb(scale(img)))
    ax[0, 2].imshow(bgr2rgb(translate(img)))
    ax[1, 0].imshow(bgr2rgb(rotate(img)))
    ax[1, 1].imshow(bgr2rgb(affine_transform(img)))
    ax[1, 2].imshow(bgr2rgb(perspective_transform(img)))
    ax[0, 0].set_title('Original')
    ax[0, 1].set_title('Scaling')
    ax[0, 2].set_title('Translation')
    ax[1, 0].set_title('Rotation')
    ax[1, 1].set_title('Affine\nTransformation')
    ax[1, 2].set_title('Perspective\nTransformation')
    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[0, 2].axis('off')
    ax[1, 0].axis('off')
    ax[1, 1].axis('off')
    ax[1, 2].axis('off')
    plt.show()

    ##############################

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(bgr2rgb(scale(translate(img))))
    ax[0, 1].imshow(bgr2rgb(scale(translate(img))))
    ax[1, 0].imshow(bgr2rgb(scale(translate(img))))
    ax[1, 1].imshow(bgr2rgb(scale(translate(img))))
    ax[0, 0].set_title('translate --> scale')
    ax[0, 1].set_title('scale --> translate')
    ax[1, 0].set_title('translate --> rotate')
    ax[1, 1].set_title('rotate --> translate')
    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[1, 0].axis('off')
    ax[1, 1].axis('off')
    plt.show()

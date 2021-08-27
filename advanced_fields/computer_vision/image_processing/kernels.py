"""
Image Processing Kernels
"""

import numpy as np
import scipy.signal as sig
import cv2
import matplotlib.pyplot as plt


def get_prewitt_kernels():
    """
    Returns the Prewitt operator kernels
    https://en.wikipedia.org/wiki/Prewitt_operator
    """
    K_prewitt_x = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])
    K_prewitt_y = np.array([[1, 1, 1],
                            [0, 0, 0],
                            [-1, -1, -1]])
    return K_prewitt_x, K_prewitt_y


def get_sobel_kernels():
    """
    Returns the Sobel operator kernels
    https://en.wikipedia.org/wiki/Sobel_operator
    """
    K_sobel_x = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])
    K_sobel_y = np.array([[1, 2, 1],
                          [0, 0, 0],
                          [-1, -2, -1]])
    return K_sobel_x, K_sobel_y


def scale(arr):
    # Note that plt.imshow() can handle the value scale well even without the scaling
    return 255 * (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def plot_two_operators(GV_x_op1, GV_y_op1, GV_x_op2, GV_y_op2,
                       op1_name='Prewitt', op2_name='Sobel'):

    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.imshow(GV_x_op1, cmap='gray')
    ax1.set_xlabel(f'Gx {op1_name}')
    ax2.imshow(GV_y_op1, cmap='gray')
    ax2.set_xlabel(f'Gy {op1_name}')
    ax3.imshow(GV_x_op2, cmap='gray')
    ax3.set_xlabel(f'Gx {op2_name}')
    ax4.imshow(GV_y_op2, cmap='gray')
    ax4.set_xlabel(f'Gy {op2_name}')

    plt.show()


##########################################

# scipy implementation

if __name__ == "__main__":
    img = cv2.imread('../../../datasets/per_field/cv/color_man_2004.jpg', cv2.IMREAD_GRAYSCALE)

    # Prewitt
    K_prewitt_x, K_prewitt_y = get_prewitt_kernels()
    GV_x_prewitt = scale(sig.convolve2d(img, K_prewitt_x[::-1, ::-1], mode='same')).astype(np.uint8)
    GV_y_prewitt = scale(sig.convolve2d(img, K_prewitt_y[::-1, ::-1], mode='same')).astype(np.uint8)

    # Sobel
    K_sobel_x, K_sobel_y = get_sobel_kernels()
    GV_x_sobel = scale(sig.convolve2d(img, K_sobel_x[::-1, ::-1], mode='same')).astype(np.uint8)
    GV_y_sobel = scale(sig.convolve2d(img, K_sobel_y[::-1, ::-1], mode='same')).astype(np.uint8)

    plot_two_operators(GV_x_prewitt, GV_y_prewitt, GV_x_sobel, GV_y_sobel)


##########################################

# cv2 implementation

if __name__ == "__main__":
    # img = cv2.imread('../../../datasets/per_field/cv/color_man_2004.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('../../../datasets/per_field/cv/color_man_2004.jpg')
    # img = np.float32(img) / 255.0  # scaling

    # Prewitt
    K_prewitt_x, K_prewitt_y = get_prewitt_kernels()
    GV_x_prewitt = cv2.filter2D(img, -1, K_prewitt_x)
    GV_y_prewitt = cv2.filter2D(img, -1, K_prewitt_y)

    # Sobel
    GV_x_sobel = scale(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)).astype(np.uint8)  # cv2.CV_8U, ksize=5
    GV_y_sobel = scale(cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)).astype(np.uint8)  # cv2.CV_8U, ksize=5

    plot_two_operators(GV_x_prewitt, GV_y_prewitt, GV_x_sobel, GV_y_sobel)

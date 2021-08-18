# https://github.com/opencv/opencv/tree/master/samples/python

from __future__ import print_function
import cv2
import numpy as np

MAX_MATCHES = 500  # MAX_FEATURES - max number of ORB features to detect in the two images.
GOOD_MATCH_PERCENT = 0.15


def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create(MAX_MATCHES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)  # We use the hamming distance as a measure of similarity between two feature descriptors
    matches = matcher.match(descriptors1, descriptors2, None)  # find the matching features in the two images, sort them by goodness of match and keep only a small percentage of original matches
    # Automatic feature matching doesn't always produce 100% accurate matches. It is not uncommon for 20-30% of the matches to be incorrect.
    matches.sort(key=lambda x: x.distance, reverse=False)  # Sort matches by score
    matches = matches[:int(len(matches) * GOOD_MATCH_PERCENT)]  # keep only top matches
    # Notice, we have many incorrect matches and thefore we will need to use a robust method to calculate homography later.

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography - calculate an accurate homography
    # A homography can be computed when we have 4 or more corresponding points in two images.
    homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC)  # findHomography() utilizes a robust estimation technique called Random Sample Consensus (RANSAC) which produces the right result even in the presence of large number of bad matches
    print("Estimated homography : \n", homography)  # Print estimated homography

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, homography, (width, height))  # Warping image. apply the transformation to all pixels in one image, to map it to the other image.

    return im1Reg


if __name__ == '__main__':
    imReference = cv2.imread("form.jpg", cv2.IMREAD_COLOR)  # Read reference\template image
    im = cv2.imread("scanned-form.jpg", cv2.IMREAD_COLOR)  # Read image to be aligned
    imReg = alignImages(im, imReference)  # Registered image will be resotred in imReg. The estimated homography will be stored in h.
    cv2.imwrite("aligned.jpg", imReg)  # Write aligned image to disk.

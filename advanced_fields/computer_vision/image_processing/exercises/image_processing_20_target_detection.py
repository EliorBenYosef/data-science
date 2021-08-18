import cv2
import numpy as np

image = cv2.imread('tarpsB.jpg', 1)
# convert to CIELab
cielab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

# define target strings
targ = ['Target 1 - Blue', 'Target 2 - Yellow', 'Target 3 - Red']
i = 0

# colors = [
#           ([91, 40, 40]), ([40, 209, 255]), ([81, 60, 166])
# ]
# rough conversion of BGR target values to CIELab
cielab_colors = [
    ([20, 20, -40]), ([80, 0, 90]), ([40, 70, 30])
]

# # loop over the boundaries
height = image.shape[0]
width = image.shape[1]
mask = np.ones(image.shape[0:2])

cv2.circle(mask, (int(width / 2), int(height / 2)), int(height / 2), 0, -1);
mask = 1 - mask
mask = mask.astype('uint8')

# for color in colors:
for cielab_color in cielab_colors:
    diff_img = cielab.astype(float)
    # find the colors within the specified boundaries and apply
    # the mask

    diff_img[:, :, 0] = np.absolute(diff_img[:, :, 0] - 255 * cielab_color[0] / 100)
    diff_img[:, :, 1] = np.absolute(diff_img[:, :, 1] - (cielab_color[1] + 128))
    diff_img[:, :, 2] = np.absolute(diff_img[:, :, 2] - (cielab_color[2] + 128))

    diff_img = (diff_img[:, :, 1] + diff_img[:, :, 2]) / 2
    diff_img = cv2.GaussianBlur(diff_img, (19, 19), 0)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(diff_img, mask)
    min_img = np.array(diff_img / 255)

    ff_mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(image, ff_mask, minLoc, 255, (12, 12, 12), (12, 12, 12), cv2.FLOODFILL_MASK_ONLY);
    ff_mask = ff_mask[1:-1, 1:-1]
    im2, contours, hierarchy = cv2.findContours(ff_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cont = contours[max_index]
    print('target color = {}'.format(image[minLoc[1], minLoc[0], :]))
    # putting text and outline rectangles on image
    x, y, w, h = cv2.boundingRect(cont)
    cv2.rectangle(image, (x, y), (x + w, y + h), colors[i], 2)
    cv2.putText(image, targ[i], (x - 50, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.85, (0, 255, 0))

    cv2.imshow('diff1D', diff_img / 255)
    cv2.imshow('ff_mask', ff_mask * 255)
    cv2.waitKey(0)
    i += 1

cv2.imshow("Show", image)
cv2.waitKey(0)

cv2.destroyAllWindows()

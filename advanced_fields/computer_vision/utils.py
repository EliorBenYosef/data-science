import numpy as np


def calculate_iou(bb1, bb2):
    """
    Calculates the Intersection Over Union (IOU) score of the two bounding-boxes
    (the overlap between two bounding boxes).
    each bounding-box coordinates are in the form of: (x_min, y_min, x_max, y_max).
    """
    bb1_x_min, bb1_y_min, bb1_x_max, bb1_y_max = bb1
    bb2_x_min, bb2_y_min, bb2_x_max, bb2_y_max = bb2

    # get the intersection's coordinate:
    intersection_x_min = max(bb1_x_min, bb2_x_min)
    intersection_x_max = min(bb1_x_max, bb2_x_max)
    intersection_y_min = max(bb1_y_min, bb2_y_min)
    intersection_y_max = min(bb1_y_max, bb2_y_max)

    # calculate the intersection's width, height, and area:
    intersection_w = max(intersection_x_max - intersection_x_min, 0)
    intersection_h = max(intersection_y_max - intersection_y_min, 0)
    intersection = intersection_w * intersection_h

    # calculate the union's area:
    union = ((bb1_x_max - bb1_x_min) * (bb1_y_max - bb1_y_min) +    # bb1 area
             (bb2_x_max - bb2_x_min) * (bb2_y_max - bb2_y_min) -    # bb2 area
             intersection)

    # calculate the IOU:
    iou = intersection / union

    return iou


def scale_0_255(X):
    """
    Rescale values back to pixel values [0, 255] for the display purpose
    Note that plt.imshow() can handle the value scale well even without the scaling
    """
    X_min, X_max = np.min(X), np.max(X)
    X_norm = (X - X_min) / (X_max - X_min)
    X_rescaled = 255 * X_norm
    return X_rescaled


##########################################

def test_iou_calculation():
    pred = (50, 50, 90, 100)
    target = (70, 80, 120, 150)
    print(calculate_iou(pred, target))

    # no intersection at all
    pred = (50, 50, 70, 100)
    target = (90, 80, 120, 150)
    print(calculate_iou(pred, target))

    # the entire pred_bb is in the target_bb
    pred = (70, 70, 80, 80)
    target = (50, 50, 100, 100)
    print(calculate_iou(pred, target))

    # the entire target_bb is in the pred_bb
    pred = (50, 50, 100, 100)
    target = (70, 70, 80, 80)
    print(calculate_iou(pred, target))

    # perfect score
    pred = (70, 70, 80, 80)
    target = (70, 70, 80, 80)
    print(calculate_iou(pred, target))


if __name__ == '__main__':
    test_iou_calculation()



import numpy as np

"""
TODO:
    i. shift box
    ii. box shrink
    iii. expand box randomly (0.8 ~ 1.2)
"""

def bounding_box_data_augmentation(bounding_boxes, imsize, percentage):
    """https://github.com/hukkelas/DeepPrivacy/blob/c1ee9983e8cbedaca6215e3f8fc695b6eb7b8bdc/deep_privacy/data_tools/dataloaders.py#L172"""
    # Data augment width and height by percentage of width.
    # Bounding box will preserve its center.
    shrink_percentage = np.random.uniform(-percentage, percentage)
    width = (bounding_boxes[2] - bounding_boxes[0]).astype(float)
    height = (bounding_boxes[3] - bounding_boxes[1]).astype(float)

    w, h = imsize

    # Can change 10% in each direction
    width_diff = shrink_percentage * width
    height_diff = shrink_percentage * height
    bounding_boxes[0] -= width_diff.astype(int)
    bounding_boxes[1] -= height_diff.astype(int)
    bounding_boxes[2] += width_diff.astype(int)
    bounding_boxes[3] += height_diff.astype(int)
    # Ensure that bounding box is within image
    bounding_boxes[0] = max(0, bounding_boxes[0])
    bounding_boxes[1] = max(0, bounding_boxes[1])
    bounding_boxes[2] = min(w, bounding_boxes[2])
    bounding_boxes[3] = min(h, bounding_boxes[3])
    return bounding_boxes

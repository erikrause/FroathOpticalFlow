import numpy as np
SMOOTH = 1e-6
def iou(outputs: np.array, labels: np.array):
    outputs = outputs//255
    labels = labels//255
    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()
    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return iou  # Or thresholded.mean()
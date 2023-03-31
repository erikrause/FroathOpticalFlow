import numpy as np
SMOOTH = 1e-6
def iou(outputs: np.array, labels: np.array):
    labels = labels//255
    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()
    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return iou  # Or thresholded.mean()
import torch
eps=1e-5
def calculate_overlap_metrics(gt, pred):
    output = pred.view(-1, )
    target = gt.view(-1, ).float()

    tp = torch.sum(output * target)  # TP
    fp = torch.sum(output * (1 - target))  # FP
    fn = torch.sum((1 - output) * target)  # FN
    tn = torch.sum((1 - output) * (1 - target))  # TN

    pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    return pixel_acc, dice, precision, specificity, recall
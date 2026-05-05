import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.manual_seed(seed)


def gen_confusion_matrix(num_class, img_predict, img_label):
    mask = (img_label != -1)
    label = num_class * img_label[mask] + img_predict[mask]
    count = torch.bincount(label, minlength=num_class ** 2)
    confusion_matrix = count.reshape(num_class, num_class)
    return confusion_matrix


def eval_metrics(confusion_matrix, mode='ts'):
    eps = 1e-7

    unique_index = np.where(np.sum(confusion_matrix, axis=1) != 0)[0]
    confusion_matrix = confusion_matrix[unique_index, :]
    confusion_matrix = confusion_matrix[:, unique_index]

    a = np.diag(confusion_matrix)
    b = np.sum(confusion_matrix, axis=0)
    c = np.sum(confusion_matrix, axis=1)

    pa = a / (c + eps)
    ua = a / (b + eps)
    f1 = 2 * pa * ua / (pa + ua + eps)
    mean_f1 = np.nanmean(f1)

    oa = np.sum(a) / np.sum(confusion_matrix)

    pe = np.sum(b * c) / (np.sum(c) * np.sum(c))
    kappa = (oa - pe) / (1 - pe)

    intersection = np.diag(confusion_matrix)
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    iou = intersection / union
    mean_iou = np.nanmean(iou)

    f1 = np.round(f1, 3)
    if mode == 'ts':
        return mean_f1, oa, kappa, mean_iou, f1
    else:
        return mean_f1

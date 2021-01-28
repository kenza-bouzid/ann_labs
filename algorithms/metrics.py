
import numpy as np


def class_precision(y, y_pred, class_label=1):
    correclty_predicted_samples_num = (
        (y_pred == class_label) & (y_pred == y)).sum()
    all_predicted_class_num = (y_pred == class_label).sum()
    return correclty_predicted_samples_num / all_predicted_class_num


def precision(y, y_pred):
    precA = class_precision(y, y_pred, -1)
    precB = class_precision(y, y_pred, 1)
    print(f'Precision for A: {precA}, for B: {precB}')
    return precA, precB


def class_recall(y, y_pred, class_label=1):
    correclty_predicted_samples_num = (
        (y_pred == class_label) & (y_pred == y)).sum()
    all_class_samples = (y == class_label).sum()
    return correclty_predicted_samples_num / all_class_samples


def recall(y, y_pred):
    recA = class_recall(y, y_pred, -1)
    recB = class_recall(y, y_pred, 1)
    print(f'Recall for A: {recA}, for B: {recB}')
    return recA, recB

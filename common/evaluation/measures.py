import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve


def make_indicator_from_probabilities(y_pred_prob,
                                      threshold):
    y_pred_indicator = np.zeros_like(y_pred_prob)
    y_pred_indicator[y_pred_prob >= threshold] = 1.0
    y_pred_indicator[y_pred_prob < threshold] = 0.0

    return y_pred_indicator


def binary_matthews_correlation_coefficient(y_true,
                                            y_pred):  # These are labels; not probabilities or logits.
    y_pos_true = y_true[:, 1]
    y_pos_pred_indicator = y_pred[:, 1]

    TP = np.count_nonzero(np.multiply(y_pos_pred_indicator,
                                      y_pos_true))
    TN = np.count_nonzero(np.multiply((y_pos_pred_indicator - 1.0),
                                      (y_pos_true - 1.0)))
    FP = np.count_nonzero(np.multiply(y_pos_pred_indicator,
                                      (y_pos_true - 1.0)))
    FN = np.count_nonzero(np.multiply((y_pos_pred_indicator - 1.0),
                                      y_pos_true))

    mcc = ((TP*TN) - (FP*FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return mcc


def get_rates_and_thresholds(y_true,
                             y_pred,
                             curve_function_type):
    measures_at_threshold = (dict(), dict())
    thresholds = dict()

    if y_true.shape[-1] != y_pred.shape[-1]:
        raise ValueError("Y true and y pred do not cover the same number of classes.")

    number_of_classes = y_true.shape[-1]

    for i in range(number_of_classes):
        if curve_function_type == "ROC":
            measures_at_threshold[0][i],\
            measures_at_threshold[1][i],\
            thresholds[i] = roc_curve(y_true[:, i],
                                      y_pred[:, i],
                                      drop_intermediate=False)
        elif curve_function_type == "PR":
            measures_at_threshold[0][i],\
            measures_at_threshold[1][i], \
            thresholds[i] = precision_recall_curve(y_true[:, i],
                                                   y_pred[:, i])
        else:
            raise ValueError("Invalid curve function type.")

    return measures_at_threshold,\
           thresholds,\
           number_of_classes


def get_optimal_threshold_per_class(measures_at_threshold,
                                    thresholds,
                                    number_of_classes,
                                    measure_function_type):
    measure_per_class = [None] * number_of_classes
    optimal_threshold_per_class = [None] * number_of_classes

    for i in range(number_of_classes):
        if measure_function_type == "J":
            measure_per_class[i] = youden_j_statistic(fpr=measures_at_threshold[0][i],
                                                      tpr=measures_at_threshold[1][i])
        elif measure_function_type == "F1":
            measure_per_class[i] = f1_measure(precision=measures_at_threshold[0][i],
                                              recall=measures_at_threshold[1][i])
        else:
            raise ValueError("Invalid measure function type.")
        optimal_threshold_per_class[i] = thresholds[i][np.argmax(measure_per_class[i])]

    return measure_per_class, optimal_threshold_per_class


def youden_j_statistic(fpr, tpr):
    return tpr - fpr


def f1_measure(precision, recall):
    return (2 * precision * recall) / (precision + recall)


def stable_softmax(X):
    exps = np.exp(X - np.max(X, 1).reshape((X.shape[0], 1)))
    return exps / np.sum(exps, 1).reshape((X.shape[0], 1))


def sigmoid(x):
    x = np.nan_to_num(x)
    return 1. / (1. + np.exp(-x))

import os
import numpy as np

def load_npys(data_dir, args):
    npys = [np.load(os.path.join(data_dir, i)) for i in args]
    if len(args) == 1:
        return npys[0]
    return npys

def mean_average_precision(Y_pred, Y_true, num_classes):
    result = 0.0
    for i in range(len(Y_pred)):
        average_precision = precision = 0.0
        true_set, pred_set = np.zeros(num_classes, np.int32), np.zeros(num_classes, np.int32)
        for j in range(max(len(Y_pred[i]), len(Y_true[i]))):
            if j < len(Y_pred[i]):
                if pred_set[Y_pred[i][j]] < true_set[Y_pred[i][j]]:
                    precision += 1
                pred_set[Y_pred[i][j]] += 1

            if j < len(Y_true[i]):
                if true_set[Y_true[i][j]] < pred_set[Y_true[i][j]]:
                    precision += 1
                true_set[Y_true[i][j]] += 1

            average_precision += precision / (j + 1)

        result += average_precision / max(len(Y_pred[i]), len(Y_true[i]))
    return result / len(Y_pred)

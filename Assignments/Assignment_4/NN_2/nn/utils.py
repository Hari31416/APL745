import numpy as np
from losses import *
from scores import *
from activations import *


def parse_loss(loss):
    if isinstance(loss, str):
        if loss == "mse":
            return MeanSquaredLoss
        elif loss == "binary_cross_entropy":
            return BinaryCrossEntropy
        elif loss == "categorical_cross_entropy":
            return CategoricalCrossEntropy
        else:
            raise ValueError("Invalid loss function")
    elif isinstance(loss, Loss):
        return loss
    else:
        raise ValueError("Invalid loss function")


def parse_metric(metric):
    if isinstance(metric, str):
        if metric == "accuracy":
            return Accuracy()
        elif metric == "mse":
            return MeanSquaredError()
        elif metric == "mae":
            return MeanAbsoluteError()
        elif metric == "precision":
            return Precision()
        elif metric == "recall":
            return Recall()
        elif metric == "f1":
            return F1()
        else:
            raise ValueError("Invalid metric function")
    elif isinstance(metric, Metric):
        return metric
    else:
        raise ValueError("Invalid metric function")


def parse_activation(activation):
    if isinstance(activation, str):
        if activation == "sigmoid":
            return Sigmoid()
        elif activation == "linear":
            return Linear()
        elif activation == "relu":
            return ReLu()
        elif activation == "softmax":
            return Softmax()
        elif activation == "tanh":
            return Tanh()
        else:
            raise ValueError("Invalid activation function")
    elif isinstance(activation, Activation):
        return activation
    else:
        raise ValueError("Invalid activation function")


def one_hot(y, n_classes):
    """
    Converts a vector of labels into a one-hot matrix.

    Parameters
    ----------
    y : array_like
        An array of shape (m, ) that contains labels for X. Each value in y
        should be an integer in the range [0, n_classes).

    n_classes : int
        The number of classes.

    Returns
    -------
    one_hot : array_like
        An array of shape (m, n_classes) where each row is a one-hot vector.
    """
    if len(y.shape) > 1:
        raise ValueError("y should be a vector")
    m = y.shape[0]
    one_hot = np.zeros((n_classes, m))
    for i in range(m):
        one_hot[y[i], i] = 1
    return one_hot

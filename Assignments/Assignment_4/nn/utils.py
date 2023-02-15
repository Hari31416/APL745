import numpy as np
from nn.losses import *
from nn.scores import *
from nn.activations import *


ALL_LOSSES = {
    "mse": MeanSquaredLoss,
    "binary_cross_entropy": BinaryCrossEntropy,
    "categorical_cross_entropy": CategoricalCrossEntropy,
}

ALL_METRICS = {
    "accuracy": Accuracy,
    "mse": MeanSquaredError,
    "mae": MeanAbsoluteError,
    "precision": Precision,
    "recall": Recall,
    "f1": F1,
}

ALL_ACTIVATIONS = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "softmax": Softmax,
}


def parse_loss(loss):
    """
    Parses the loss function from string to Loss object

    Parameters
    ----------
    loss : str or Loss
        The loss function

    Raises
    ------
    ValueError
        If the activation function is not a string or Loss object
    Returns
    -------
    Loss
    """
    if isinstance(loss, str):
        if loss in ALL_LOSSES:
            return ALL_LOSSES[loss]
        else:
            raise ValueError(
                "Invalid loss function. Please use one of the following: {}".format(
                    ALL_LOSSES.keys()
                )
            )
    elif isinstance(loss, Loss):
        return loss
    else:
        raise ValueError("Invalid loss function")


def parse_metric(metric):
    """
    Parses the metric function from string to Metric object

    Parameters
    ----------
    metric : str or Metric
        The metric function

    Raises
    ------
    ValueError
        If the activation function is not a string or Metric object
    Returns
    -------
    Metric
    """
    if isinstance(metric, str):
        if metric in ALL_METRICS:
            return ALL_METRICS[metric]()
        else:
            raise ValueError(
                "Invalid metric function. Please use one of the following: {}".format(
                    ALL_METRICS.keys()
                )
            )
    elif isinstance(metric, Metric):
        return metric
    else:
        raise ValueError("Invalid metric function")


def parse_activation(activation):
    """
    Parses the activation function from string to Activation object

    Parameters
    ----------
    activation : str or Activation
        The activation function

    Raises
    ------
    ValueError
        If the activation function is not a string or Activation object

    Returns
    -------
    Activation
    """
    if isinstance(activation, str):
        if activation in ALL_ACTIVATIONS:
            return ALL_ACTIVATIONS[activation]()
        else:
            raise ValueError(
                "Invalid activation function. Please use one of the following: {}".format(
                    ALL_ACTIVATIONS.keys()
                )
            )
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

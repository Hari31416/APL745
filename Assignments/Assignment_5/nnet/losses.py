import numpy as np


class Loss:
    "The base class for loss"

    def __init__(self) -> None:
        pass

    def loss(self):
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __str__(self) -> str:
        return self.__repr__()


class MeanSquaredLoss(Loss):
    def __init__(self) -> None:
        super().__init__()

    def loss(self, y_true: np.ndarray, y_hat: np.ndarray) -> float:

        _, m = y_true.shape
        return np.sum(np.square((y_true - y_hat)) / (2 * m))

    def derivative(self, y_true: np.ndarray, y_hat: np.ndarray):

        _, m = y_true.shape
        d = (y_hat - y_true) / m
        return d


class BinaryCrossEntropy(Loss):
    def __init__(self) -> None:
        super().__init__()

    def loss(self, y_true: np.ndarray, y_hat: np.ndarray) -> float:

        _, m = y_true.shape
        term_1 = y_true * np.log(y_hat + 1e-10)
        term_2 = (1 - y_true) * np.log(1 - y_hat + 1e-10)
        return -np.sum(term_1 + term_2) / m

    def derivative(self, y_true: np.ndarray, y_hat: np.ndarray):

        _, m = y_true.shape
        res = -(y_true / (y_hat + 1e-10) - (1 - y_true) / (1 - y_hat + 1e-10))
        return res / m


class CategoricalCrossEntropy(Loss):
    def __init__(self) -> None:
        super().__init__()

    def loss(self, y_true: np.ndarray, y_hat: np.ndarray) -> float:

        _, m = y_true.shape
        return -np.sum(y_true * np.log(y_hat + 1e-10)) / m

    def derivative(self, y_true: np.ndarray, y_hat: np.ndarray):
        _, m = y_true.shape
        return (y_hat - y_true) / m

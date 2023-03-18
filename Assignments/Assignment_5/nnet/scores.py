import numpy as np


class Metric:
    "The base class for all scores"

    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__

    def score(self):
        pass


class ClassificationMetric(Metric):
    def __init__(self) -> None:
        super().__init__()

    def score(self, y_true: np.ndarray, y_hat: np.ndarray) -> float:
        pass


class RegressionMetric(Metric):
    def __init__(self) -> None:
        super().__init__()

    def score(self, y_true: np.ndarray, y_hat: np.ndarray) -> float:
        pass


class Accuracy(ClassificationMetric):
    def __init__(self) -> None:
        super().__init__()

    def score(self, y_true: np.ndarray, y_hat: np.ndarray) -> float:
        if y_hat.shape[0] != 1:
            raise ValueError("y_hat should be a vector")
        if y_true.shape[0] != 1:
            raise ValueError("y_true should be a vector")
        return np.mean(y_true == y_hat)


class MeanSquaredError(RegressionMetric):
    def __init__(self) -> None:
        super().__init__()

    def score(self, y_true: np.ndarray, y_hat: np.ndarray) -> float:
        m = y_true.shape[-1]
        return np.sum(np.square((y_true - y_hat)) / (2 * m))


class MeanAbsoluteError(RegressionMetric):
    def __init__(self) -> None:
        super().__init__()

    def score(self, y_true: np.ndarray, y_hat: np.ndarray) -> float:
        m = y_true.shape[-1]
        return np.sum(np.abs(y_true - y_hat)) / m


class Precision(ClassificationMetric):
    def __init__(self) -> None:
        super().__init__()

    def score(self, y_true: np.ndarray, y_hat: np.ndarray) -> float:
        return np.sum(np.logical_and(y_true == 1, y_hat == 1)) / (
            np.sum(y_hat == 1) + 1
        )


class Recall(ClassificationMetric):
    def __init__(self) -> None:
        super().__init__()

    def score(self, y_true: np.ndarray, y_hat: np.ndarray) -> float:
        return np.sum(np.logical_and(y_true == 1, y_hat == 1)) / np.sum(y_true == 1)


class F1(ClassificationMetric):
    def __init__(self) -> None:
        super().__init__()

    def score(self, y_true: np.ndarray, y_hat: np.ndarray) -> float:
        precision = np.sum(np.logical_and(y_true == 1, y_hat == 1)) / np.sum(y_hat == 1)
        recall = np.sum(np.logical_and(y_true == 1, y_hat == 1)) / np.sum(y_true == 1)
        return 2 * precision * recall / (precision + recall)

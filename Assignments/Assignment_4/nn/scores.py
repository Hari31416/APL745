import numpy as np


class Metric:
    "The base class for all scores"

    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__

    def score(self, y_true: np.ndarray, y_hat: np.ndarray):
        """
        Calculate the score

        Parameters
        ----------
        y_true : np.ndarray
            The true values Label encoded
        y_hat : np.ndarray
            The predicted values Label encoded

        Returns
        -------
        float
            The score
        """
        pass


class ClassificationMetric(Metric):
    """
    The base class for all classification scores
    """

    def __init__(self) -> None:
        super().__init__()

    def score(self, y_true: np.ndarray, y_hat: np.ndarray) -> float:
        pass


class RegressionMetric(Metric):
    """
    The base class for all regression scores
    """

    def __init__(self) -> None:
        super().__init__()

    def score(self, y_true: np.ndarray, y_hat: np.ndarray) -> float:
        pass


class MeanSquaredError(RegressionMetric):
    """
    The mean squared error score
    Defined as: 1/2 * sum((y_true - y_hat)^2)
    """

    def __init__(self) -> None:
        """
        The mean squared error score
        Defined as: 1/2 * sum((y_true - y_hat)^2)
        """
        super().__init__()

    def score(self, y_true: np.ndarray, y_hat: np.ndarray) -> float:
        m = y_true.shape[-1]
        return np.sum(np.square((y_true - y_hat)) / (2 * m))


class MeanAbsoluteError(RegressionMetric):
    """
    The mean absolute error score
    Defined as: 1/m * sum(|y_true - y_hat|)
    """

    def __init__(self) -> None:
        """
        The mean absolute error score
        Defined as: 1/m * sum(|y_true - y_hat|)
        """
        super().__init__()

    def score(self, y_true: np.ndarray, y_hat: np.ndarray) -> float:
        m = y_true.shape[-1]
        return np.sum(np.abs(y_true - y_hat)) / m


class Accuracy(ClassificationMetric):
    """
    The accuracy score
    Defined as: sum(y_true == y_hat) / m
    """

    def __init__(self) -> None:
        """
        The accuracy score
        Defined as: sum(y_true == y_hat) / m
        """
        super().__init__()

    def score(self, y_true: np.ndarray, y_hat: np.ndarray) -> float:
        if y_hat.shape[0] != 1:
            raise ValueError("y_hat should be a vector")
        if y_true.shape[0] != 1:
            raise ValueError("y_true should be a vector")
        return np.mean(y_true == y_hat)


class Precision(ClassificationMetric):
    """
    The precision score
    Defined as: sum(y_true == 1 and y_hat == 1) / sum(y_hat == 1)
    """

    def __init__(self) -> None:
        """
        The precision score
        Defined as: sum(y_true == 1 and y_hat == 1) / sum(y_hat == 1)
        """
        super().__init__()

    def score(self, y_true: np.ndarray, y_hat: np.ndarray) -> float:
        return np.sum(np.logical_and(y_true == 1, y_hat == 1)) / (
            np.sum(y_hat == 1) + 1
        )


class Recall(ClassificationMetric):
    """
    Recall score
    Defined as: sum(y_true == 1 and y_hat == 1) / sum(y_true == 1)
    """

    def __init__(self) -> None:
        """
        Recall score
        Defined as: sum(y_true == 1 and y_hat == 1) / sum(y_true == 1)
        """
        super().__init__()

    def score(self, y_true: np.ndarray, y_hat: np.ndarray) -> float:
        return np.sum(np.logical_and(y_true == 1, y_hat == 1)) / np.sum(y_true == 1)


class F1(ClassificationMetric):
    """
    F1 score
    Defined as: 2 * precision * recall / (precision + recall)
    """

    def __init__(self) -> None:
        """
        F1 score
        Defined as: 2 * precision * recall / (precision + recall)
        """
        super().__init__()

    def score(self, y_true: np.ndarray, y_hat: np.ndarray) -> float:
        precision = np.sum(np.logical_and(y_true == 1, y_hat == 1)) / np.sum(y_hat == 1)
        recall = np.sum(np.logical_and(y_true == 1, y_hat == 1)) / np.sum(y_true == 1)
        return 2 * precision * recall / (precision + recall)

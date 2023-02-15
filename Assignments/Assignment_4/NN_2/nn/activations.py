import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
    """
    The base class for all activations
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, input: np.ndarray) -> np.ndarray:
        pass


class Sigmoid(Activation):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, input: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-input))

    def derivative(self, input: np.ndarray) -> np.ndarray:
        res = self(input) * (1 - self(input))
        return res


class Linear(Activation):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, input: np.ndarray) -> np.ndarray:
        return input

    def derivative(self, input: np.ndarray) -> np.ndarray:
        return np.ones(input.shape)


class ReLu(Activation):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, input: np.ndarray) -> np.ndarray:
        out = np.maximum(0, input)
        return out

    def derivative(self, input: np.ndarray) -> np.ndarray:
        return np.where(input > 0, 1, 0)


class Tanh(Activation):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, input: np.ndarray) -> np.ndarray:
        return np.tanh(input)

    def derivative(self, input: np.ndarray) -> np.ndarray:
        return 1 - np.square(self(input))


class Softmax(Activation):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, input: np.ndarray) -> np.ndarray:
        exps = np.exp(input - np.max(input))
        return exps / (np.sum(exps, axis=0, keepdims=True))

    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of the softmax function

        Parameters
        ----------
        input: np.ndarray
            The input to the softmax function. This is Z = Wx + b

        Returns
        -------
        np.ndarray
            The derivative of the softmax function
        """
        diff = np.ones(input.shape)
        return diff

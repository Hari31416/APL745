import numpy as np
from abc import ABC, abstractmethod
from nn.activations import *
from nn.utils import *


class Layer(ABC):
    """
    An abstract base class for Layers
    """

    def __init__(self):
        self.input = None
        self.output = None
        self.weight = None
        self.bias = None
        self.dW = None
        self.db = None

    def __repr__(self) -> str:
        return "Layer"

    def __str__(self) -> str:
        return self.__repr__()

    @abstractmethod
    def forward(self, input_data):
        pass

    @abstractmethod
    def backward(self):
        pass

    def __repr__(self) -> str:
        return "Layer"

    def update(self, lr):
        pass


class Input(Layer):
    """
    A class for input layer. Provides the input shape.
    """

    def __init__(self, input_shape, name="Input"):
        super().__init__()
        self.input_shape = input_shape
        self.name = name

    def __repr__(self) -> str:
        return f"Input({self.input_shape})"

    def forward(self, input_data):
        """
        The forward pass of the Input layer
        This layer does not do anything except storing the input data

        Parameters
        ----------
        input_data : np.ndarray
            The input data

        Returns
        -------
        np.ndarray
            The output of the layer
        """
        self.input = input_data
        self.output = input_data
        return self.input

    def backward(self):
        """
        The backward pass of the Input layer
        This layer does not do anything as it is the Input layer
        """
        pass


class Dense(Layer):
    """
    The fully connected layer
    """

    def __init__(self, neurons, activation="sigmoid", name=None, l1=0, l2=0):
        """
        Initialize the dense layer

        Parameters
        ----------
        neurons : int
            The number of neurons in the layer
        activation : str, optional
            The activation function, by default "sigmoid"
        name : str, optional
            The name of the layer, by default None
        l1 : float, optional
            The l1 regularization parameter, by default 0
        l2 : float, optional
            The l2 regularization parameter, by default 0

        Raises
        ------
        ValueError
            If the activation function is not supported
            If the name is duplicated

        Returns
        -------
        Dense
            The dense layer
        """
        super().__init__()
        self.activation = parse_activation(activation)
        self.neurons = neurons
        self.name = name
        self.weight = None
        self.bias = None
        self.dW = None
        self.db = None
        self.l1 = l1
        self.l2 = l2

    def __repr__(self) -> str:
        return f"Dense({self.neurons})"

    def forward(self, input):
        """
        Calculate the forward pass of the layer

        Parameters
        ----------
        input : np.ndarray
            The input data

        Returns
        -------
        np.ndarray
            The output of the layer
        """
        self.input = input
        Z = np.dot(self.weight, input) + self.bias
        self.Z = Z
        A = self.activation(Z)
        self.output = A
        return A

    def backward(self, delta_l):
        """
        Do the backward pass of the layer

        Parameters
        ----------
        delta_l : np.ndarray
            The delta of the current layer

        Raises
        ------
        ValueError
            If the dW or db is exploding

        Returns
        -------
        np.ndarray
            The delta of the previous layer
        """
        l1_loss = self.l1 * np.sign(self.weight) / self.input.shape[1]
        l2_loss = self.l2 * self.weight / self.input.shape[1]
        delta_next = delta_l * self.activation.derivative(self.Z)
        dW = (
            np.dot(delta_next, self.input.T)
            + l1_loss * np.sign(self.weight)
            + l2_loss * self.weight
        )
        db = np.sum(delta_next, axis=1, keepdims=True)
        assert dW.shape == self.weight.shape
        assert db.shape == self.bias.shape
        if dW.max() > 1000:
            raise ValueError("dW is Exploding", dW.max(), dW.shape)
        if db.max() > 1000:
            raise ValueError("db is Exploding", db.max(), db.shape)
        self.dW = dW
        self.db = db
        delta_next = np.dot(self.weight.T, delta_next)
        return delta_next

    def update(self, lr):
        """
        Update the weights and biases of the layer

        Parameters
        ----------
        lr : float
            The learning rate

        Returns
        -------
        None
        """
        self.weight -= lr * self.dW
        self.bias -= lr * self.db


class Dropout(Layer):
    """
    The dropout layer
    """

    def __init__(self, rate=0.5, name=None):
        """
        The dropout layer

        Parameters
        ----------
        rate : float, optional
            The dropout rate, by default 0.5
        name : str, optional
            The name of the layer, by default None

        Returns
        -------
        Dropout
        """
        super().__init__()
        self.rate = rate
        self.name = name
        self.mask = None

    def __repr__(self) -> str:
        return f"Dropout({self.rate})"

    def forward(self, input):
        """
        Forward pass of the dropout layer

        Parameters
        ----------
        input : np.ndarray
            The input data

        Returns
        -------
        np.ndarray
            The output of the layer
        """
        self.input = input
        self.mask = np.random.rand(*input.shape) < (1 - self.rate)
        self.output = self.input * self.mask
        return self.output

    def backward(self, delta_l):
        """
        Backward pass of the dropout layer

        Parameters
        ----------
        delta_l : np.ndarray
            The delta of the current layer

        Returns
        -------
        np.ndarray
            The delta of the previous layer
        """
        delta_next = delta_l * self.mask
        return delta_next


# TODO: Implement the batch normalization layer
class BatchNormalization(Layer):
    """
    The batch normalization layer
    """

    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.gamma = None
        self.beta = None
        self.dgamma = None
        self.dbeta = None
        self.mean = None
        self.variance = None
        self.epsilon = 1e-8
        self.x_norm = None
        self.x_hat = None
        self.x_mu = None

    def __repr__(self) -> str:
        return f"BatchNormalization()"

    def forward(self, input):
        # TODO: Implement the forward pass
        self.input = input
        # self.mean = np.mean(input, axis=0)
        # self.variance = np.var(input, axis=0)
        # self.x_hat = (input - self.mean) / np.sqrt(self.variance + self.epsilon)
        # self.x_norm = self.gamma * self.x_hat + self.beta
        # self.output = self.x_norm
        # return self.output
        self.output = self.input
        return self.output

    def backward(self, delta_l):
        # TODO: Implement the backward pass
        # m = self.input.shape[1]
        # self.dgamma = np.sum(delta_l * self.x_hat, axis=0)
        # self.dbeta = np.sum(delta_l, axis=0)
        # dx_hat = delta_l * self.gamma
        # dx_mu1 = dx_hat / np.sqrt(self.variance + self.epsilon)
        # dvar = np.sum(
        #     dx_hat
        #     * (self.input - self.mean)
        #     * -0.5
        #     * (self.variance + self.epsilon) ** (-1.5),
        #     axis=0,
        # )
        # dmu = np.sum(
        #     dx_hat * -1 / np.sqrt(self.variance + self.epsilon), axis=0
        # ) + dvar * np.mean(-2 * (self.input - self.mean), axis=0)
        # dx = dx_mu1 + dvar * 2 * (self.input - self.mean) / m + dmu / m
        # return dx
        return delta_l

    # def update(self, lr):
    # TODO: Implement the update step
    #     self.gamma -= lr * self.dgamma
    #     self.beta -= lr * self.dbeta

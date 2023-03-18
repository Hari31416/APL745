import numpy as np
from dl.nn.activations import *
from utils import optimizers
from dl.nn.helper import *
from copy import deepcopy


class Layer:

    """
    Base class for all the layers in the neural network.
    """

    def __init__(self):
        """
        Initialize the input and output for the layer.
        """
        self.input = None
        self.output = None
        self.input_shape = 0
        self.units = 0
        self.initializer = None
        self.optimizer_w = optimizers.Adam()
        self.optimizer_b = optimizers.Adam()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def forward_propagation(self, input):

        """
        Implement forward propagation through the layer.

        Parameters
        ----------
        input: numpy.ndarray, shape (n_[l-1], batch_size)
            Input data to be propagated through the layer.

        Returns
        -------
        output: numpy.ndarray, shape (n_[l], batch_size)
            Output of the layer after forward propagation.

        """

        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):

        """
        Implement backward propagation through the layer.

        Parameters
        ----------

        output_error: numpy.ndarray
            Error in the output of the layer.

        learning_rate: float
            Learning rate to be used for weight updates.

        Returns
        -------
        numpy.ndarray
            Error to be propagated back to the previous layer.
        """

        raise NotImplementedError


class BatchNormalization(Layer):

    """
    Class for batch normalization layer in the neural network.

    Parameters
    ----------

    n_l: int
        Number of neurons in the output layer.
    """

    def __init__(self, units):

        """
        Initialize the batch normalization layer with weights and biases.

        Parameters
        ----------
        n_l: int
            Number of neurons in the output layer.
        """

        super().__init__()
        self.gamma = np.ones((units, 1))  # gamma (n_[l], 1)
        self.beta = np.zeros((units, 1))  # beta  (n_[l], 1)
        self.epsilon = 1e-8
        self.units = units

    def forward_propagation(self, activate_l_1):

        """
        Implement forward propagation through the batch normalization layer.

        Parameters
        ----------
        activate_l_1: numpy.ndarray
            activate_l_1 data to be propagated through the layer.

        Returns
        -------
        numpy.ndarray
            z of the layer after forward propagation.
        """

        self.input = activate_l_1  # activate (n_[l], batch_size)
        self.mean = np.mean(self.input, axis=1, keepdims=True)  # mean (n_[l], 1)
        self.variance = np.var(self.input, axis=1, keepdims=True)  # variance (n_[l], 1)
        self.std = np.sqrt(self.variance + self.epsilon)  # std (n_[l], 1)
        self.z = (self.input - self.mean) / self.std  # z (n_[l], batch_size)
        self.output = self.gamma * self.z + self.beta  # output (n_[l], batch_size)

        return self.output  # output (n_[l], batch_size)

    def backward_propagation(self, output_error):

        """
        Implement backward propagation through the batch normalization layer.

        Parameters
        ----------
        output_error: numpy.ndarray
            Error in the output of the layer.

        Returns
        -------
        numpy.ndarray
            Error to be propagated back to the previous layer.
        """

        self.dgamma = np.sum(output_error * self.z, axis=1, keepdims=True)
        self.dbeta = np.sum(output_error, axis=1, keepdims=True)
        self.dz = output_error * self.gamma
        self.dvariance = np.sum(
            self.dz
            * (self.input - self.mean)
            * (-0.5)
            * (self.variance + self.epsilon) ** (-1.5),
            axis=1,
            keepdims=True,
        )
        self.dmean = (
            np.sum(self.dz * (-1) / self.std, axis=1, keepdims=True)
            + self.dvariance
            * np.sum(-2 * (self.input - self.mean), axis=1, keepdims=True)
            / self.input.shape[1]
        )
        self.dinput = (
            self.dz / self.std
            + self.dvariance * 2 * (self.input - self.mean) / self.input.shape[1]
            + self.dmean / self.input.shape[1]
        )

        self.gamma -= self.optimizer_w.update(self.dgamma)
        self.beta -= self.optimizer_b.update(self.dbeta)

        return self.dinput


class DenseLayer(Layer):

    """
    Class for fully connected layer in the neural network.

    Parameters
    ----------

    n_l_1: int
        Number of neurons in the input layer.

    n_l: int
        Number of neurons in the output layer.
    """

    def __init__(
        self, input_shape=None, units=None, activation=None, l1=0, l2=0, use_bias=True
    ):

        """
        Initialize the fully connected layer with weights and biases.

        Parameters
        ----------
        n_l_1: int
            Number of neurons in the input layer.

        n_l: int
            Number of neurons in the output layer.

        activation: str, optional
            Activation function to be used in the layer.
            for adding user defined activation function. use the ActivationLayer .

        l1: float, optional
            L1 regularization parameter.

        l2: float, optional
            L2 regularization parameter.

        use_bias: bool, optional
            Whether to use bias in the layer or not.
        """

        super().__init__()

        if input_shape is not None:
            self.input_shape = input_shape  # (n_[l-1], batch_size)

        # set the number of neurons in this layer
        self.units = units  # n_[l]

        # whether to use bias in the layer or not
        self.use_bias = use_bias

        # set the regularization parameters
        self.l1 = l1
        self.l2 = l2

        # set the activation function for the layer if specified by the user
        if activation is not None:
            if activation.lower() not in [
                "sigmoid",
                "tanh",
                "relu",
                "softmax",
                "linear",
                "hard_sigmoid",
            ]:
                raise ValueError("Invalid activation function.")
            else:
                if activation.lower() == "linear":
                    self.activation = ActivationLayer(Linear)
                elif activation.lower() == "hard_sigmoid":
                    self.activation = ActivationLayer(HardSigmoid)
                elif activation.lower() == "softmax":
                    self.activation = ActivationLayer(Softmax)
                elif activation.lower() == "sigmoid":
                    self.activation = ActivationLayer(Sigmoid)
                elif activation.lower() == "tanh":
                    self.activation = ActivationLayer(Tanh)
                elif activation.lower() == "relu":
                    self.activation = ActivationLayer(ReLU)

        else:
            self.activation = None

    def initialize(self, optimizer=optimizers.Adam(), initializer="glorot_normal"):
        """
        Initialize the optimizer for the layer.

        Parameters
        ----------
        optimizer: Optimizer
            Optimizer to be used for updating the weights and biases.

        initializer: str, optional
            Weight initializer to be used for initializing the weights.

        """
        self.optimizer_w = deepcopy(optimizer)

        self.weights = initializers(
            method=initializer, shape=(self.units, self.input_shape)
        )
        if self.use_bias:
            self.biases = initializers(method=initializer, shape=(self.units, 1))
            self.optimizer_b = deepcopy(optimizer)
        else:
            self.biases = 0
        # include bias parameters only if use_bias is True

        self.trainable_params = self.weights.size + self.units * self.use_bias

    def forward_propagation(self, activate_l_1):

        """
        Implement forward propagation through the fully connected layer.

        Parameters
        ----------
        activate_l_1: numpy.ndarray
            activate_l_1 data to be propagated through the layer.

        Returns
        -------
        numpy.ndarray
            z of the layer after forward propagation.
        """

        self.input = activate_l_1  # activate (n_[l], batch_size)
        z_l = (
            np.dot(self.weights, self.input) + self.biases
        )  # z = weights x input + biases = (n_[l], n_[l-1]) x (n_[l-1], batch_size) + (n_[l], 1) = (n_[l], batch_size)

        # Apply activation function if present
        if self.activation is not None:
            z_l = self.activation.forward_propagation(z_l)

        self.output = z_l

        return self.output  # z (n_[l], batch_size)

    def backward_propagation(self, output_error):

        """
        Implement backward propagation through the fully connected layer.

        Parameters
        ----------
        output_error: numpy.ndarray
            Error in the output of the layer.

        Returns
        -------
        numpy.ndarray
            Error to be propagated back to the previous layer.
        """

        # Apply activation function if present in the dense layer itself
        if self.activation is not None:
            output_error = self.activation.backward_propagation(output_error)

        # calculating the error with respect to weights before updating the weights
        input_error = np.dot(
            self.weights.T, output_error
        )  # weights x output_error  (n_[l], n_[l-1]) x (n_[l], batch_size) = (n_[l-1], batch_size)
        weights_error = np.dot(
            output_error, self.input.T
        )  # output_error x input    (n_l, batch_size) x (n_[l-1], batch_size) = (n_[l], n_[l-1])

        m_samples = output_error.shape[1]

        # addition of regularization term, by default both l1 and l2 are 0
        reg_term = (self.l1 / m_samples) * np.sign(self.weights)
        reg_term = (self.l2 / m_samples) * self.weights

        self.weights -= self.optimizer_w.update(weights_error + reg_term)

        if self.use_bias:
            self.biases -= np.sum(
                self.optimizer_b.update(output_error), axis=1, keepdims=True
            )

        return input_error


class ActivationLayer(Layer):

    """
    Activation layer for neural networks.
    This layer applies an non-linearity to the input data.

    Parameters:
    -----------

    activation (class) : (callable)
        The class of the activation function to be used. The class should have two methods:
        activation and activation_prime.
        activation:
            The activation function.

        activation_prime:
            The derivative of the activation function.

    """

    def __init__(self, activation):
        super().__init__()
        self.activation = activation.activation
        self.activation_prime = activation.activation_prime
        self.activation_name = activation.__name__

    def forward_propagation(self, z_l):

        """
        Perform the forward propagation of the activation layer.

        Parameters:
        z (numpy.ndarray): The z to the layer.

        Returns:
        numpy.ndarray: The output of the layer after applying the activation function.
        """

        self.input = z_l
        activate_l = self.activation(self.input)
        self.output = activate_l

        return self.output  # (n_[l], batch_size)

    def backward_propagation(self, output_error):

        """
        Perform the backward propagation of the activation layer.

        Parameters:
        -----------

        output_error (numpy.ndarray):
            The error that needs to be backpropagated through the layer.

        Returns:
        --------

        numpy.ndarray:
            The input error after backward propagation through the activation layer.
        """
        # if self.activation_prime == None:
        #     return output_error

        del_J_del_A_l = np.multiply(
            self.activation_prime(self.input), output_error
        )  # element-wise multiplication (n_[l], batch_size) x (n_[l], batch_size) = (n_[l], batch_size)

        return del_J_del_A_l


class DropoutLayer(Layer):

    """
    Dropout layer for neural networks.
    This layer randomly drops out units during training to prevent overfitting.

    Parameters:
    ----------

    dropout_rate (float):
        The dropout rate. The rate of neurons that will be dropped out.
        It should be a float in the range of [0, 1].

    """

    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mask = None
        self.trainable_params = 0

    def forward_propagation(self, input):

        """
        Perform the forward propagation of the dropout layer.

        Parameters:
        -----------

        input (numpy.ndarray):
            The input to the layer.

        Returns:
        --------

        numpy.ndarray:
            The output of the layer after applying dropout.
        """

        self.input = input  # input (n_[l], batch_size)

        # sample from a binomial distribution with p = 1 - dropout_rate
        self.mask = np.random.binomial(
            1, 1 - self.dropout_rate, size=input.shape
        )  # mask (n_[l], batch_size)

        self.output = (self.input * self.mask) / (
            1 - self.dropout_rate
        )  # (n_[l], batch_size)
        return self.output  # (n_[l], batch_size)

    def backward_propagation(self, output_error):

        """
        Perform the backward propagation of the dropout layer.

        Parameters:
        -----------

        output_error (numpy.ndarray):
            The error that needs to be backpropagated through the layer.

        Returns:
        --------

        numpy.ndarray:
            The input error after backward propagation through the dropout layer.
        """

        return (output_error * self.mask) / (
            1 - self.dropout_rate
        )  # (n_[l], batch_size)


class FlattenLayer(Layer):

    """
    Flatten layer for 2D inputs.

    Parameters:
    -----------
    None

    Attributes:
    -----------
    None
    """

    def __init__(self) -> None:
        super().__init__()

    def forward_propagation(self, inputs):
        """
        Perform forward propagation for the flatten layer.

        Parameters:
        -----------
        inputs: numpy.ndarray
            The input data with shape (height, width, channels, samples).

        Returns:
        --------
        output: numpy.ndarray
            The output data after flattening, with shape ( height * width * channels, samples).
        """

        self.input_height, self.input_width, self.channels, self.samples = inputs.shape

        # Flatten the input
        output = inputs.reshape(
            self.input_height * self.input_width * self.channels, self.samples
        )

        return output

    def backward_propagation(self, dL_doutput):
        """
        Perform backward propagation for the flatten layer.

        Parameters:
        -----------
        dL_doutput: numpy.ndarray
            The gradient of the loss with respect to the output data.

        Returns:
        --------
        dL_dinputs: numpy.ndarray
            The gradient of the loss with respect to the input data.
        """

        # Reshape the gradient of the loss with respect to the output data
        dL_dinputs = dL_doutput.reshape(
            self.input_height, self.input_width, self.channels, self.samples
        )

        return dL_dinputs


class ConvLayer2(Layer):
    def __init__(
        self,
        input_shape=None,
        filters=5,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding
        self.use_bias = use_bias
        self.trainable_params = 0

        if input_shape is not None:
            self.input_shape = input_shape  # shape (height, width, channels)

        self.l1 = 0
        self.l2 = 0

    def _get_shapes(self):
        self.input_height, self.input_width, self.channels = self.input_shape
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.strides

        if self.padding == "valid":
            pad_top = 0
            pad_bottom = 0
            pad_left = 0
            pad_right = 0

        elif self.padding == "same":
            pad_top = (kernel_height - 1) // 2
            pad_bottom = kernel_height - 1 - pad_top
            pad_left = (kernel_width - 1) // 2
            pad_right = kernel_width - 1 - pad_left

        self.pad_top = pad_top
        self.pad_bottom = pad_bottom
        self.pad_left = pad_left
        self.pad_right = pad_right

        # Calculate the output shape
        output_height = (
            self.input_height + pad_top + pad_bottom - kernel_height
        ) // stride_height + 1
        output_width = (
            self.input_width + pad_left + pad_right - kernel_width
        ) // stride_width + 1

        self.output_shape = (output_height, output_width, self.filters)

    def initialize(self, optimizer=optimizers.Adam(), initializer="glorot_uniform"):
        self.optimizer_w = deepcopy(optimizer)

        self.weights = initializers(
            method=initializer,
            shape=(
                self.kernel_size[0],
                self.kernel_size[1],
                self.input_shape[2],
                self.filters,
            ),
        ) / (
            self.kernel_size[0] * self.kernel_size[1]
        )  # shape (kernel_height, kernel_width, prev_layer_filters, filters)

        if self.use_bias:
            self.biases = np.zeros((self.filters, 1))
            self.optimizer_b = deepcopy(optimizer)
        else:
            self.biases = 0

        self.trainable_params = self.weights.size + self.use_bias * self.filters

        self._get_shapes()

    def __convolve(self, images, kernel, strides, output_shape):
        """
        Convolve the images with the kernel using the given strides.

        Parameters:
        -----------
        images: numpy.ndarray
            The input images with shape (height, width, channels, batch_size).
        kernel: numpy.ndarray
            The kernel with shape (kernel_height, kernel_width, prev_layer_filters, filters).
        strides: tuple
            The strides with shape (stride_height, stride_width).
        output_shape: tuple
            The shape of the output with shape (output_height, output_width, filters, batch_size).

        Returns:
        --------
        output: numpy.ndarray
            The output of the convolution with shape (output_height, output_width, filters, batch_size).
        """
        output = np.zeros(output_shape)
        out_h, out_w, _, samples = output_shape
        # image_height, image_width, _, batch_size = images.shape
        kernel_height, kernel_width, _, n_filters = kernel.shape
        stride_height, stride_width = strides

        for image_ in range(samples):
            single_image = images[:, :, :, image_]
            for height_start in range(0, out_h, stride_height):
                for width_start in range(0, out_w, stride_width):
                    for n_filter in range(n_filters):
                        output[height_start, width_start, n_filter, image_] = np.sum(
                            single_image[
                                height_start : height_start + kernel_height,
                                width_start : width_start + kernel_width,
                                :,
                            ]
                            * kernel[:, :, :, n_filter]
                        )

        return output

    def forward_propagation(self, inputs):
        """
        Perform forward propagation for the convolutional layer.

        Parameters:
        -----------
        inputs: numpy.ndarray
            The input data with shape (height, width, channels, batch_size).

        Returns:
        --------
        output: numpy.ndarray
            The output data after convolution, with shape ( output_height, output_width, filters, batch_size).
        """

        # Pad the input
        inputs = np.pad(
            inputs,
            (
                (self.pad_top, self.pad_bottom),
                (self.pad_left, self.pad_right),
                (0, 0),
                (0, 0),
            ),
            mode="constant",
        )

        self.input = inputs

        output_shape = self.output_shape + (inputs.shape[-1],)

        # Convolve the input with the kernel
        output = self.__convolve(inputs, self.weights, self.strides, output_shape)

        # Add the bias
        if self.use_bias:
            output += self.biases

        return output

    def backward_propagation(self, dL_doutput):
        """
        Perform backward propagation for the convolutional layer.

        Parameters:
        -----------
        dL_doutput: numpy.ndarray shape (output_height, output_width, filters, batch_size)
            The gradient of the loss with respect to the output data.

        Returns:
        --------
        dL_dinputs: numpy.ndarray
            The gradient of the loss with respect to the input data.
        """
        input = self.input
        _, _, _, samples = dL_doutput.shape

        # # Pad the input
        # input = np.pad(input, ((self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right), (0, 0), (0, 0)), mode="constant")

        # initialize the input gradient with zeros to send to the previous layer
        dL_dinput = np.zeros(input.shape)

        # initialize the kernel gradient with zeros to update the kernel
        dL_dkernel = np.zeros(self.weights.shape)

        # initialize the bias gradient with zeros to update the bias
        if self.use_bias:
            dL_dbias = np.zeros(self.biases.shape)

        # Calculate the gradient of the loss with respect to the input
        for image_ in range(samples):
            single_dL = dL_doutput[:, :, :, image_]
            single_image_ = input[:, :, :, image_]
            for height_start in range(0, dL_doutput.shape[0], self.strides[0]):
                for width_start in range(0, dL_doutput.shape[1], self.strides[1]):
                    for n_filter in range(self.filters):
                        fliped_kernel = np.flip(
                            self.weights[:, :, :, n_filter], axis=(0, 1)
                        )
                        dL_dinput[
                            height_start : height_start + self.kernel_size[0],
                            width_start : width_start + self.kernel_size[1],
                            :,
                            image_,
                        ] += (
                            fliped_kernel
                            * single_dL[height_start, width_start, n_filter]
                        )
                        dL_dkernel[:, :, :, n_filter] += (
                            single_image_[
                                height_start : height_start + self.kernel_size[0],
                                width_start : width_start + self.kernel_size[1],
                                :,
                            ]
                            * single_dL[height_start, width_start, n_filter]
                        )

        # Calculate the gradient of the loss with respect to the kernel
        # for height_start in range(0, dL_doutput.shape[0], self.strides[0]):
        #     for width_start in range(0, dL_doutput.shape[1], self.strides[1]):
        #         for n_filter in range(self.filters):
        #             dL_dkernel[:, :, :, n_filter] += np.sum(input[height_start:height_start + self.kernel_size[0], width_start:width_start + self.kernel_size[1], :, :] * dL_doutput[height_start, width_start, n_filter, np.newaxis], axis=3)

        # Calculate the gradient of the loss with respect to the bias
        # dL_dbias = np.sum(dL_doutput, axis=(0, 1, 3))

        # Update the kernel and bias
        self.weights -= self.optimizer_w.update(dL_dkernel)
        if self.use_bias:
            self.biases -= self.optimizer_b.update(dL_dbias)

        # Remove the padding from the input gradient
        dL_dinput = dL_dinput[
            self.pad_top : dL_dinput.shape[0] - self.pad_bottom,
            self.pad_left : dL_dinput.shape[1] - self.pad_right,
            :,
            :,
        ]

        return dL_dinput


class ConvLayer(Layer):
    def __init__(
        self,
        input_shape=None,
        filters=5,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding
        self.use_bias = use_bias
        self.trainable_params = 0

        if input_shape is not None:
            self.input_shape = input_shape  # shape (height, width, channels)

        self.l1 = 0
        self.l2 = 0

    def _get_shapes(self):
        self.input_height, self.input_width, self.channels = self.input_shape
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.strides

        if self.padding == "valid":
            pad_top = 0
            pad_bottom = 0
            pad_left = 0
            pad_right = 0

        elif self.padding == "same":
            pad_top = (kernel_height - 1) // 2
            pad_bottom = kernel_height - 1 - pad_top
            pad_left = (kernel_width - 1) // 2
            pad_right = kernel_width - 1 - pad_left

        self.pad_top = pad_top
        self.pad_bottom = pad_bottom
        self.pad_left = pad_left
        self.pad_right = pad_right

        # Calculate the output shape
        output_height = (
            self.input_height + pad_top + pad_bottom - kernel_height
        ) // stride_height + 1
        output_width = (
            self.input_width + pad_left + pad_right - kernel_width
        ) // stride_width + 1

        self.output_shape = (output_height, output_width, self.filters)

    def initialize(self, optimizer=optimizers.Adam(), initializer="glorot"):
        self.optimizer_w = deepcopy(optimizer)

        self.weights = initializers(
            method=initializer,
            shape=(
                self.kernel_size[0],
                self.kernel_size[1],
                self.input_shape[2],
                self.filters,
            ),
        )  # shape (kernel_height, kernel_width, prev_layer_filters, filters)

        if self.use_bias:
            self.biases = np.zeros((self.filters, 1))
            self.optimizer_b = deepcopy(optimizer)
        else:
            self.biases = 0

        self.trainable_params = self.weights.size + self.use_bias * self.filters

        self._get_shapes()

    def __convolve(self, images, kernel, strides, output_shape):
        """
        Convolve the images with the kernel using the given strides.

        Parameters:
        -----------
        images: numpy.ndarray
            The input images with shape (height, width, channels, batch_size).
        kernel: numpy.ndarray
            The kernel with shape (kernel_height, kernel_width, prev_layer_filters, filters).
        strides: tuple
            The strides with shape (stride_height, stride_width).
        output_shape: tuple
            The shape of the output with shape (output_height, output_width, filters, batch_size).

        Returns:
        --------
        output: numpy.ndarray
            The output of the convolution with shape (output_height, output_width, filters, batch_size).
        """
        output = np.zeros(output_shape)
        out_h, out_w, _, samples = output_shape
        # image_height, image_width, _, batch_size = images.shape
        kernel_height, kernel_width, _, n_filters = kernel.shape
        stride_height, stride_width = strides

        for n_filter in range(n_filters):
            for height_start in range(0, out_h, stride_height):
                for width_start in range(0, out_w, stride_width):
                    output[height_start, width_start, n_filter, :] = np.sum(
                        images[
                            height_start : height_start + kernel_height,
                            width_start : width_start + kernel_width,
                            :,
                            :,
                        ]
                        * kernel[:, :, :, n_filter, np.newaxis],
                        axis=(0, 1, 2),
                    )  # sum aaccross the height, width and channels not the batch size

            output[:, :, n_filter, :] += self.biases[n_filter]

        return output

    def forward_propagation(self, inputs):
        """
        Perform forward propagation for the convolutional layer.

        Parameters:
        -----------
        inputs: numpy.ndarray
            The input data with shape (height, width, channels, batch_size).

        Returns:
        --------
        output: numpy.ndarray
            The output data after convolution, with shape ( output_height, output_width, filters, batch_size).
        """

        # Pad the input
        inputs = np.pad(
            inputs,
            (
                (self.pad_top, self.pad_bottom),
                (self.pad_left, self.pad_right),
                (0, 0),
                (0, 0),
            ),
            mode="constant",
        )

        self.input = inputs

        output_shape = self.output_shape + (inputs.shape[-1],)

        # Convolve the input with the kernel
        output = self.__convolve(inputs, self.weights, self.strides, output_shape)

        # Add the bias
        if self.use_bias:
            output += self.biases

        return output

    def backward_propagation(self, dL_doutput):
        """
        Perform backward propagation for the convolutional layer.

        Parameters:
        -----------
        dL_doutput: numpy.ndarray shape (output_height, output_width, filters, batch_size)
            The gradient of the loss with respect to the output data.

        Returns:
        --------
        dL_dinputs: numpy.ndarray
            The gradient of the loss with respect to the input data.
        """
        input = self.input

        # # Pad the input
        # input = np.pad(input, ((self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right), (0, 0), (0, 0)), mode="constant")

        # initialize the input gradient with zeros to send to the previous layer
        dL_dinput = np.zeros(input.shape)

        # initialize the kernel gradient with zeros to update the kernel
        dL_dkernel = np.zeros(self.weights.shape)
        dL_dbias = np.zeros(self.biases.shape)

        # initialize the bias gradient with zeros to update the bias
        if self.use_bias:
            dL_dbias = np.zeros(self.biases.shape)

        # Calculate the gradient of the loss with respect to the input
        for n_filter in range(self.filters):
            for height_start in range(0, dL_doutput.shape[0], self.strides[0]):
                for width_start in range(0, dL_doutput.shape[1], self.strides[1]):

                    dL_dinput[
                        height_start : height_start + self.kernel_size[0],
                        width_start : width_start + self.kernel_size[1],
                        :,
                        :,
                    ] += (
                        self.weights[:, :, :, n_filter, np.newaxis]
                        * dL_doutput[height_start, width_start, n_filter, :]
                    )
                    dL_dkernel[:, :, :, n_filter] += np.sum(
                        input[
                            height_start : height_start + self.kernel_size[0],
                            width_start : width_start + self.kernel_size[1],
                            :,
                            :,
                        ]
                        * dL_doutput[height_start, width_start, n_filter, :],
                        axis=3,
                    )
                    dL_dbias[n_filter] += np.sum(
                        dL_doutput[height_start, width_start, n_filter, :]
                    )

        # Calculate the gradient of the loss with respect to the kernel
        # for height_start in range(0, dL_doutput.shape[0], self.strides[0]):
        #     for width_start in range(0, dL_doutput.shape[1], self.strides[1]):
        #         for n_filter in range(self.filters):
        #             dL_dkernel[:, :, :, n_filter] += np.sum(input[height_start:height_start + self.kernel_size[0], width_start:width_start + self.kernel_size[1], :, :] * dL_doutput[height_start, width_start, n_filter, np.newaxis], axis=3)

        # Calculate the gradient of the loss with respect to the bias
        # dL_dbias = np.sum(dL_doutput, axis=(0, 1, 3))

        # Update the kernel and bias
        self.weights -= self.optimizer_w.update(dL_dkernel)
        if self.use_bias:
            self.biases -= self.optimizer_b.update(dL_dbias)

        # Remove the padding from the input gradient
        dL_dinput = dL_dinput[
            self.pad_top : dL_dinput.shape[0] - self.pad_bottom,
            self.pad_left : dL_dinput.shape[1] - self.pad_right,
            :,
            :,
        ]

        return dL_dinput


class MaxPool2D(Layer):
    def __init__(self, pool_size, strides):
        """
        Initialize the max pooling layer.

        Parameters:
        -----------
        pool_size: tuple
            The size of the pooling window with shape (pool_height, pool_width).
        strides: tuple
            The strides of the pooling window with shape (stride_height, stride_width).
        """
        self.pool_size = (
            pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        )
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.input_shape = None  # shape (height, width, channels)

    def _get_shapes(self):
        self.input_height, self.input_width, self.channels = self.input_shape
        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.strides

        self.output_height = (self.input_height - pool_height) // stride_height + 1
        self.output_width = (self.input_width - pool_width) // stride_width + 1

        self.output_shape = (self.output_height, self.output_width, self.channels)

    def forward_propagation(self, inputs):
        """
        Perform forward propagation for the max pooling layer.

        Parameters:
        -----------
        inputs: numpy.ndarray
            The input data with shape (height, width, channels, batch_size).

        Returns:
        --------
        output: numpy.ndarray
            The output data after max pooling, with shape (output_height, output_width, channels, batch_size).
        """
        self.input = inputs

        output_shape = self.output_shape + (inputs.shape[3],)

        # Initialize the output
        output = np.zeros(output_shape)

        # Loop through the input and pool
        for image_ in range(inputs.shape[-1]):
            for height_start in range(0, output_shape[0], self.strides[0]):
                for width_start in range(0, output_shape[1], self.strides[1]):
                    for channel in range(inputs.shape[2]):
                        output[height_start, width_start, channel, image_] = np.max(
                            inputs[
                                height_start : height_start + self.pool_size[0],
                                width_start : width_start + self.pool_size[1],
                                channel,
                                image_,
                            ]
                        )

        return output

    def backward_propagation(self, dL_doutput):
        """
        Perform backward propagation for the max pooling layer.

        Parameters:
        -----------
        dL_doutput: numpy.ndarray
            The gradient of the loss with respect to the output data.

        Returns:
        --------
        dL_dinput: numpy.ndarray
            The gradient of the loss with respect to the input data.
        """
        input = self.input

        # Initialize the gradient of the loss with respect to the input with zeros
        dL_dinput = np.zeros(input.shape)

        # Loop through the input and pool
        for image_ in range(input.shape[-1]):
            for height_start in range(0, dL_doutput.shape[0], self.strides[0]):
                for width_start in range(0, dL_doutput.shape[1], self.strides[1]):
                    for channel in range(input.shape[2]):
                        # Find the index of the maximum value
                        max_index = np.argmax(
                            input[
                                height_start : height_start + self.pool_size[0],
                                width_start : width_start + self.pool_size[1],
                                channel,
                                image_,
                            ]
                        )

                        # Calculate the indices of the maximum value in the input
                        height_index, width_index = np.unravel_index(
                            max_index, self.pool_size
                        )

                        # Update the gradient of the loss with respect to the input
                        dL_dinput[
                            height_start + height_index,
                            width_start + width_index,
                            channel,
                            image_,
                        ] += dL_doutput[height_start, width_start, channel, image_]

        return dL_dinput


class MaxPool2D_(Layer):
    def __init__(self, pool_size, strides):
        """
        Initialize the max pooling layer.

        Parameters:
        -----------
        pool_size: tuple
            The size of the pooling window with shape (pool_height, pool_width).
        strides: tuple
            The strides of the pooling window with shape (stride_height, stride_width).
        """
        self.pool_size = (
            pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        )
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.input_shape = None  # shape (height, width, channels)

    def _get_shapes(self):
        self.input_height, self.input_width, self.channels = self.input_shape
        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.strides

        self.output_height = (self.input_height - pool_height) // stride_height + 1
        self.output_width = (self.input_width - pool_width) // stride_width + 1

        self.output_shape = (self.output_height, self.output_width, self.channels)

    def forward_propagation(self, inputs):
        """
        Perform forward propagation for the max pooling layer.

        Parameters:
        -----------
        inputs: numpy.ndarray
            The input data with shape (height, width, channels, batch_size).

        Returns:
        --------
        output: numpy.ndarray
            The output data after max pooling, with shape (output_height, output_width, channels, batch_size).
        """
        self.input = inputs

        output_shape = self.output_shape + (inputs.shape[3],)

        # Initialize the output
        output = np.zeros(output_shape)

        # Loop through the input and pool
        # for image_ in range(inputs.shape[-1]):
        for height_start in range(0, output_shape[0], self.strides[0]):
            for width_start in range(0, output_shape[1], self.strides[1]):
                # for channel in range(inputs.shape[2]):
                output[height_start, width_start, :, :] = np.max(
                    inputs[
                        height_start : height_start + self.pool_size[0],
                        width_start : width_start + self.pool_size[1],
                        :,
                        :,
                    ],
                    axis=(0, 1),
                )

        return output

    def backward_propagation(self, dL_doutput):
        """
        Perform backward propagation for the max pooling layer.

        Parameters:
        -----------
        dL_doutput: numpy.ndarray
            The gradient of the loss with respect to the output data.

        Returns:
        --------
        dL_dinput: numpy.ndarray
            The gradient of the loss with respect to the input data.
        """
        input = self.input

        # Initialize the gradient of the loss with respect to the input with zeros
        dL_dinput = np.zeros(input.shape)

        mask = np.zeros(dL_doutput.shape)
        for i in range(dL_doutput.shape[0]):
            for j in range(dL_doutput.shape[1]):
                # for k in range(input.shape[2]):
                mask[i, j, :, :] = mask[i, j, :, :] == np.max(
                    input[i, j, :, :], axis=(0, 1)
                )

        dL_dinput = dL_doutput * mask

        # Loop through the input and pool
        # for image_ in range(input.shape[-1]):
        # for height_start in range(0, dL_doutput.shape[0], self.strides[0]):
        #     for width_start in range(0, dL_doutput.shape[1], self.strides[1]):
        #         # for channel in range(input.shape[2]):
        #             # Find the index of the maximum value
        #         max_index = np.argmax(
        #             input[
        #                 height_start : height_start + self.pool_size[0],
        #                 width_start : width_start + self.pool_size[1],
        #                 :,
        #                 :,
        #             ], axis=(0, 1)
        #         )

        #         # Calculate the indices of the maximum value in the input
        #         height_index, width_index = np.unravel_index(
        #             max_index, self.pool_size
        #         )

        #         # Update the gradient of the loss with respect to the input
        #         dL_dinput[
        #             height_start + height_index,
        #             width_start + width_index,
        #             :,
        #             :,
        #         ] += dL_doutput[height_start, width_start, :, :]

        return dL_dinput

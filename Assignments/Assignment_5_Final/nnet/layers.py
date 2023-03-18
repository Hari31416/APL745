from nnet.activations import *
from nnet.utils import *
from nnet.convolution import *

import numpy as np
from abc import ABC, abstractmethod


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
        self.trainable = False

    def __repr__(self) -> str:
        return f"Input({self.input_shape})"

    def forward(self, input_data):
        self.input = input_data
        self.output = input_data
        return self.input

    def backward(self):
        pass

    def _output_shape(self, input_shape):
        return input_shape

    def _calc_W_b_shape(self, input_shape, output_shape):
        return None, None

    def _num_params(self, W_shape, b_shape):
        return 0


class Dense(Layer):
    """
    The dense layer
    """

    def __init__(self, neurons, activation="sigmoid", name=None, l1=0, l2=0):
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
        self.input = input
        Z = np.dot(self.weight, input) + self.bias
        self.Z = Z
        A = self.activation(Z)
        self.output = A
        return A

    def backward(self, delta_l):
        l1_loss = self.l1 * np.sign(self.weight)
        l2_loss = self.l2 * self.weight
        delta_next = delta_l * self.activation.derivative(self.Z)
        dW = (
            np.dot(delta_next, self.input.T)
            + l1_loss * np.sign(self.weight)
            + l2_loss * self.weight
        )
        db = np.sum(delta_next, axis=1, keepdims=True)
        assert dW.shape == self.weight.shape
        assert db.shape == self.bias.shape
        if dW.max() > 500:
            raise ValueError("dW is Exploding", dW.max(), dW.shape)
        if db.max() > 500:
            raise ValueError("db is Exploding", db.max(), db.shape)
        self.dW = dW
        self.db = db
        delta_next = np.dot(self.weight.T, delta_next)
        return delta_next

    def update(self, lr):
        self.weight -= lr * self.dW
        self.bias -= lr * self.db

    def _output_shape(self, input_shape):
        return (self.neurons,)

    def _calc_W_b_shape(self, input_shape, output_shape):
        return (output_shape[0], input_shape[0]), (output_shape[0], 1)

    def _num_params(self, W_shape, b_shape):
        return np.prod(W_shape) + np.prod(b_shape)


class Dropout(Layer):
    """
    The dropout layer
    """

    def __init__(self, rate=0.5, name=None):
        super().__init__()
        self.rate = rate
        self.name = name
        self.mask = None

    def __repr__(self) -> str:
        return f"Dropout({self.rate})"

    def forward(self, input):
        self.input = input
        self.mask = np.random.rand(*input.shape) < (1 - self.rate)
        self.output = self.input * self.mask
        return self.output

    def backward(self, delta_l):
        delta_next = delta_l * self.mask
        return delta_next

    def _output_shape(self, input_shape):
        return input_shape

    def _calc_W_b_shape(self, input_shape, output_shape):
        return None, None

    def _num_params(self, W_shape, b_shape):
        return 0


class Flatten(Layer):
    """
    The flatten layer
    """

    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def __repr__(self) -> str:
        return f"Flatten()"

    def forward(self, input):
        self.input = input
        m = input.shape[-1]
        self.output = input.reshape((np.prod(input.shape[:-1]), m))
        return self.output

    def backward(self, delta_l):
        delta_next = delta_l.reshape(self.input.shape)
        return delta_next

    def _output_shape(self, input_shape):
        return (np.prod(input_shape),)

    def _calc_W_b_shape(self, input_shape, output_shape):
        return None, None

    def _num_params(self, W_shape, b_shape):
        return 0


class Reshape(Layer):
    """
    The reshape layer
    """

    def __init__(self, output_shape, name=None):
        super().__init__()
        self.output_shape = output_shape
        self.name = name

    def __repr__(self) -> str:
        return f"Reshape({self.output_shape})"

    def forward(self, input):
        self.input = input
        m = input.shape[-1]
        output = np.zeros(
            (self.output_shape[0], self.output_shape[1], self.output_shape[2], m)
        )
        for i in range(m):
            output[:, :, :, i] = input[:, i].reshape(self.output_shape)
        self.output = output
        return self.output

    def backward(self, delta_l):
        delta_next = delta_l.reshape(self.input.shape)
        return delta_next

    def update(self, lr):
        pass

    def _output_shape(self, input_shape):
        return self.output_shape

    def _calc_W_b_shape(self, input_shape, output_shape):
        return None, None

    def _num_params(self, W_shape, b_shape):
        return 0


class ConvLayer2(Layer):
    def __init__(
        self,
        filters=5,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        activation="relu",
        name=None,
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
        self.activation = activation
        self.name = name
        self.conv = Convolution()

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

    def forward(self, input):
        """
        Perform forward propagation for the convolutional layer.

        Parameters:
        -----------
        input: numpy.ndarray
            The input data with shape (height, width, channels, batch_size).

        Returns:
        --------
        output: numpy.ndarray
            The output data after convolution, with shape ( output_height, output_width, filters, batch_size).
        """

        # Pad the input
        input = np.pad(
            input,
            (
                (self.pad_top, self.pad_bottom),
                (self.pad_left, self.pad_right),
                (0, 0),
                (0, 0),
            ),
            mode="constant",
        )

        self.input = input

        output_shape = self.output_shape + (input.shape[-1],)

        # Convolve the input with the kernel
        self.Z = self.__convolve(input, self.weights, self.strides, output_shape)
        self.output = self.activation(self.Z)
        return self.output

    def backward(self, delta_l):
        """
        Perform backward propagation for the convolutional layer.

        Parameters:
        -----------
        delta_l: numpy.ndarray shape (output_height, output_width, filters, batch_size)
            The gradient of the loss with respect to the output data.

        Returns:
        --------
        dL_dinputs: numpy.ndarray
            The gradient of the loss with respect to the input data.
        """
        input = self.input
        _, _, _, samples = delta_l.shape

        # initialize the input gradient with zeros to send to the previous layer
        dL_dinput = np.zeros(input.shape)

        # initialize the kernel gradient with zeros to update the kernel
        dL_dkernel = np.zeros(self.weights.shape)

        # initialize the bias gradient with zeros to update the bias
        dL_dbias = np.zeros(self.biases.shape)

        # Calculate the gradient of the loss with respect to the input
        for image_ in range(samples):
            single_dL = delta_l[:, :, :, image_]
            single_image_ = input[:, :, :, image_]
            for height_start in range(0, delta_l.shape[0], self.strides[0]):
                for width_start in range(0, delta_l.shape[1], self.strides[1]):
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

        self.dW = dL_dkernel
        self.db = dL_dbias
        # Remove the padding from the input gradient
        dL_dinput = dL_dinput[
            self.pad_top : dL_dinput.shape[0] - self.pad_bottom,
            self.pad_left : dL_dinput.shape[1] - self.pad_right,
            :,
            :,
        ]
        delta_next = dL_dinput

        return delta_next

    def update(self, lr):
        """
        Update the weights and biases of the layer.

        Parameters:
        -----------
        lr: float
            The learning rate.
        """
        self.weights -= lr * self.dW
        self.biases -= lr * self.db

    def _output_shape(self, input_shape):
        size = self.conv._output_shape(
            input_shape,
            self.kernel_size,
            self.strides,
            self.padding,
            self.filters,
        )
        return size

    def _calc_W_b_shape(self, input_shape, output_shape):
        hi, wi, ci = input_shape
        ho, wo, co = output_shape
        W_shape = (self.kernel_height, self.kernel_width, ci, co)
        b_shape = (co, 1)
        return W_shape, b_shape

    def _num_params(self, W_shape, b_shape):
        return np.prod(W_shape) + np.prod(b_shape)


class MaxPool2D(Layer):
    def __init__(
        self,
        pool_size=2,
        strides=2,
        name=None,
    ):
        super().__init__()
        self.pool_size = (
            pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        )
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.kernel_height, self.kernel_width = pool_size
        self.stride = self.strides[0]
        self.name = name
        self.pool = Convolution()

    def __repr__(self) -> str:
        return f"MaxPool2D({self.pool_size}, {self.stride})"

    def forward(self, input):
        """
        Perform forward propagation for the max pooling layer.

        Parameters:
        -----------
        input: numpy.ndarray
            The input data with shape (height, width, channels, batch_size).

        Returns:
        --------
        output: numpy.ndarray
            The output data after max pooling, with shape (output_height, output_width, channels, batch_size).
        """
        self.input = input

        output_shape = self.output_shape + (input.shape[3],)

        # Initialize the output
        output = np.zeros(output_shape)

        # Loop through the input and pool
        for image_ in range(input.shape[-1]):
            for height_start in range(0, output_shape[0], self.strides[0]):
                for width_start in range(0, output_shape[1], self.strides[1]):
                    for channel in range(input.shape[2]):
                        output[height_start, width_start, channel, image_] = np.max(
                            input[
                                height_start : height_start + self.pool_size[0],
                                width_start : width_start + self.pool_size[1],
                                channel,
                                image_,
                            ]
                        )

        return output

    def backward(self, delta_l):
        """
        Perform backward propagation for the max pooling layer.

        Parameters:
        -----------
        delta_l: numpy.ndarray
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
            for height_start in range(0, delta_l.shape[0], self.strides[0]):
                for width_start in range(0, delta_l.shape[1], self.strides[1]):
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
                        ] += delta_l[height_start, width_start, channel, image_]
        delta_next = dL_dinput
        return delta_next

    def _output_shape(self, input_shape):
        hi, wi, ci = input_shape
        ho = (hi - self.kernel_height) // self.stride + 1
        wo = (wi - self.kernel_width) // self.stride + 1
        self.output_shape = (ho, wo, ci)
        return (
            ho,
            wo,
            ci,
        )

    def _calc_W_b_shape(self, input_shape, output_shape):
        return None, None

    def _num_params(self, W_shape, b_shape):
        return 0


class Conv2D(Layer):
    """
    The Convolutional layer
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(2, 2),
        padding="same",
        activation="relu",
        name=None,
    ):
        super().__init__()
        self.filters = filters
        # self.kernel_size = kernel_size
        # self.kernel_height, self.kernel_width = kernel_size
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        # self.strides = stride
        self.padding = padding
        self.activation = parse_activation(activation)
        self.name = name
        self.weight = None
        self.bias = None
        self.dW = None
        self.db = None
        self.conv = Convolution()

        self.kernel_height, self.kernel_width = self.kernel_size
        self.stride_height, self.stride_width = self.strides

        if self.padding == "valid":
            pad_top = 0
            pad_bottom = 0
            pad_left = 0
            pad_right = 0

        elif self.padding == "same":
            pad_top = (self.kernel_height - 1) // 2
            pad_bottom = self.kernel_height - 1 - pad_top
            pad_left = (self.kernel_width - 1) // 2
            pad_right = self.kernel_width - 1 - pad_left

        self.pad_top = pad_top
        self.pad_bottom = pad_bottom
        self.pad_left = pad_left
        self.pad_right = pad_right

    def __repr__(self) -> str:
        return (
            f"Conv2D({self.filters}, {self.kernel_size}, {self.stride}, {self.padding})"
        )

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

    def forward(self, input):
        self.input = input
        self.Z = self.conv.convolve(
            input=input,
            kernels=self.weight,
            bias=self.bias,
            stride=self.strides[0],
            padding=self.padding,
        )
        self.output = self.activation(self.Z)
        return self.output

    def backward(self, delta_l):
        input = self.input
        _, _, _, samples = delta_l.shape

        # initialize the input gradient with zeros to send to the previous layer
        dL_dinput = np.zeros(input.shape)

        # initialize the kernel gradient with zeros to update the kernel
        dL_dkernel = np.zeros(self.weight.shape)

        # initialize the bias gradient with zeros to update the bias
        dL_dbias = np.zeros(self.bias.shape)

        # Calculate the gradient of the loss with respect to the input
        for image_ in range(samples):
            single_dL = delta_l[:, :, :, image_]
            single_image_ = input[:, :, :, image_]
            for height_start in range(0, delta_l.shape[0], self.strides[0]):
                for width_start in range(0, delta_l.shape[1], self.strides[1]):
                    for n_filter in range(self.filters):
                        fliped_kernel = np.flip(
                            self.weight[:, :, :, n_filter], axis=(0, 1)
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

        self.dW = dL_dkernel
        self.db = dL_dbias
        # Remove the padding from the input gradient
        dL_dinput = dL_dinput[
            self.pad_top : dL_dinput.shape[0] - self.pad_bottom,
            self.pad_left : dL_dinput.shape[1] - self.pad_right,
            :,
            :,
        ]
        delta_next = dL_dinput

        return delta_next

    def update(self, lr):
        self.weight -= lr * self.dW
        self.bias -= lr * self.db

    def _output_shape(self, input_shape):
        size = self.conv._output_shape(
            input_shape,
            self.kernel_size,
            self.strides[0],
            self.padding,
            self.filters,
        )
        return size

    def _calc_W_b_shape(self, input_shape, output_shape):
        hi, wi, ci = input_shape
        ho, wo, co = output_shape
        W_shape = (self.kernel_height, self.kernel_width, ci, co)
        b_shape = (co, 1)
        return W_shape, b_shape

    def _num_params(self, W_shape, b_shape):
        return np.prod(W_shape) + np.prod(b_shape)


class Conv2D_(Layer):
    """
    The Convolutional layer
    """

    def __init__(
        self,
        filters,
        kernel_size,
        stride=1,
        padding="same",
        activation="relu",
        name=None,
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_height, self.kernel_width = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = parse_activation(activation)
        self.name = name
        self.weight = None
        self.bias = None
        self.dW = None
        self.db = None
        self.conv = Convolution()

    def __repr__(self) -> str:
        return (
            f"Conv2D({self.filters}, {self.kernel_size}, {self.stride}, {self.padding})"
        )

    def forward(self, input):
        self.input = input
        self.Z = self.conv.convolve(
            input=input,
            kernels=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        self.output = self.activation(self.Z)
        return self.output

    def backward(self, delta_l):
        delta_next = self.conv.convolve_backward(
            delta_l,
            self.weight,
            self.stride,
            self.padding,
        )
        delta_next = delta_next * self.activation.derivative(self.output)
        self.dW = self.conv.convolve_filter_backward(
            delta_l,
            self.input,
            self.stride,
            self.padding,
        )
        self.db = np.sum(delta_l, axis=(0, 2, 3), keepdims=True)
        return delta_next

    def update(self, lr):
        self.weight -= lr * self.dW
        self.bias -= lr * self.db

    def _output_shape(self, input_shape):
        size = self.conv._output_shape(
            input_shape,
            self.kernel_size,
            self.stride,
            self.padding,
            self.filters,
        )
        return size

    def _calc_W_b_shape(self, input_shape, output_shape):
        hi, wi, ci = input_shape
        ho, wo, co = output_shape
        W_shape = (self.kernel_height, self.kernel_width, ci, co)
        b_shape = (co, 1)
        return W_shape, b_shape

    def _num_params(self, W_shape, b_shape):
        return np.prod(W_shape) + np.prod(b_shape)


class MaxPool2D_(Layer):
    """
    The Max Pooling layer
    """

    def __init__(
        self,
        kernel_size=2,
        stride=2,
        name=None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel_height, self.kernel_width = kernel_size
        self.stride = stride
        self.name = name
        self.pool = Convolution()

    def __repr__(self) -> str:
        return f"MaxPool2D({self.kernel_size}, {self.stride})"

    def forward(self, input):
        self.input = input
        self.output = self.pool.max_pool(
            input=input, kernel_size=self.kernel_size, stride=self.stride
        )
        return self.output

    def backward(self, delta_l):
        delta_next = self.pool.max_pool_backward(
            delta_l,
            self.input,
            self.kernel_size,
            self.stride,
        )
        return delta_next

    def _output_shape(self, input_shape):
        hi, wi, ci = input_shape
        ho = (hi - self.kernel_height) // self.stride + 1
        wo = (wi - self.kernel_width) // self.stride + 1
        return (
            ho,
            wo,
            ci,
        )

    def _calc_W_b_shape(self, input_shape, output_shape):
        return None, None

    def _num_params(self, W_shape, b_shape):
        return 0


class AveragePool2D(Layer):
    """
    The Average Pooling layer
    """

    def __init__(
        self,
        kernel_size=2,
        stride=2,
        name=None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.name = name
        self.pool = Convolution()

    def __repr__(self) -> str:
        return f"AveragePool2D({self.kernel_size}, {self.stride})"

    def forward(self, input):
        self.input = input
        self.output = self.pool.average_pool(
            input=input, kernel_size=self.kernel_size, stride=self.stride
        )
        return self.output

    def backward(self, delta_l):
        delta_next = self.pool.average_pool_backward(
            delta_l, self.input, self.kernel_size, self.stride
        )
        return delta_next

    def _output_shape(self, input_shape):
        hi, wi, ci = input_shape
        ho = (hi - self.kernel_size) // self.stride + 1
        wo = (wi - self.kernel_size) // self.stride + 1
        return (
            ho,
            wo,
            ci,
        )

    def _calc_W_b_shape(self, input_shape, output_shape):
        return None, None

    def _num_params(self, W_shape, b_shape):
        return 0

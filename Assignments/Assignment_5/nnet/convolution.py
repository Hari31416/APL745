import numpy as np
from scipy.signal import correlate as corr


class Convolution:
    def __init__(self) -> None:
        pass

    def __checks(self, input, kernels):
        if len(input.shape) != 3:
            raise ValueError(
                f"The input must be of shape (H, W, C) where H is the height, W is the width and C is the number of channels. Got: {input.shape}"
            )
        if len(kernels.shape) != 4:
            raise ValueError(
                f"The kernel must be of shape (H, W, C, K) where H is the height, W is the width, C is the number of channels and K is the number of kernels. Got: {kernels.shape}"
            )
        if input.shape[-1] != kernels.shape[-2]:
            raise ValueError(
                f"The number of channels in the input and kernel must be the same: {input.shape[-1]} != {kernels.shape[-2]}"
            )
        return None

    def __parse_padding(self, padding, kernel):
        hk, hw = kernel
        if padding == "same":
            padding_h = int((hk - 1) / 2)
            padding_w = int((hw - 1) / 2)
        elif padding == "valid":
            padding_h = 0
            padding_w = 0
        elif padding == "full":
            padding_h = int(hk - 1)
            padding_w = int(hw - 1)
        else:
            raise ValueError(f"Invalid padding: {padding}")
        return padding_h, padding_w

    def _output_shape(self, input_shape, kernel, stride, padding, num_filters):
        padding = self.__parse_padding(padding, kernel)
        hi, wi, ci = input_shape
        hk, wk = kernel
        ho = (hi - hk + 2 * padding[0]) // stride + 1
        wo = (wi - wk + 2 * padding[1]) // stride + 1
        return (
            ho,
            wo,
            num_filters,
        )

    def convolve(self, input, kernels, bias, stride=1, padding="valid"):
        """
        Calculate the convolution of an input with a kernel. Works with any number of channels.

        The function is slow because it uses a for loop to iterate over the output volume.

        Parameters
        ----------
        input : numpy.ndarray
            The input to convolve shape (H, W, C)
        kernel : numpy.ndarray
            The kernel to convolve with shape (H, W, C)
        stride : int
            The stride of the convolution, default is 1
        padding : str or int
            The padding of the convolution, default is 'same'
            - If 'same' the padding is calculated to keep the output size the same as the input size
            - If 'valid' no padding is applied
            - If 'full' the padding is calculated to keep the output size the same as the input size + the kernel size - 1
            - If int, a padding of that size is applied

        Returns
        -------
        numpy.ndarray
            The convolved input, shape (H, W)
        """
        output_shape = self._output_shape(
            input[:, :, :, 0].shape,
            kernels.shape[:2],
            stride,
            padding,
            kernels.shape[-1],
        )
        ho, wo, kco = output_shape
        mi = input.shape[-1]

        output = np.zeros((ho, wo, kco, mi))
        for i in range(mi):
            output[:, :, :, i] = self.convolve_one(
                input[:, :, :, i], kernels, bias, stride, padding
            )
        return output

    def convolve_one(self, input, kernels, bias, stride=1, padding="valid"):
        """
        Calculate the convolution of an input with a kernel. Works with any number of channels.

        The function is slow because it uses a for loop to iterate over the output volume.

        Parameters
        ----------
        input : numpy.ndarray
            The input to convolve shape (H, W, C)
        kernel : numpy.ndarray
            The kernel to convolve with shape (H, W, C)
        stride : int
            The stride of the convolution, default is 1
        padding : str or int
            The padding of the convolution, default is 'same'
            - If 'same' the padding is calculated to keep the output size the same as the input size
            - If 'valid' no padding is applied
            - If 'full' the padding is calculated to keep the output size the same as the input size + the kernel size - 1
            - If int, a padding of that size is applied

        Returns
        -------
        numpy.ndarray
            The convolved input, shape (H, W)
        """
        self.__checks(input, kernels)
        hi, wi, ci = input.shape
        hk, wk, kci, kco = kernels.shape

        padding = self.__parse_padding(padding, (hk, wk))
        ho = int((hi - hk + 2 * padding[0])) + 1
        wo = int((wi - wk + 2 * padding[1])) + 1
        input = np.pad(
            input,
            ((padding[0], padding[1]), (padding[0], padding[1]), (0, 0)),
            "constant",
        )

        output = np.zeros((ho, wo, kco))
        for i in range(kco):
            feature_map = (
                self._convolve_fast(input, kernels[:, :, :, i], (ho, wo)) + bias[i]
            )
            output[:, :, i] = feature_map

        # account for strides
        if stride > 1:
            output = output[::stride, ::stride, :]
        return output

    def _convolve_slow(self, input, kernel, stride=1, padding="same"):
        """
        Calculate the convolution of an input with a kernel. Works with any number of channels.

        The function is slow because it uses a for loop to iterate over the output volume.

        Parameters
        ----------
        input : numpy.ndarray
            The input to convolve shape (H, W, C)
        kernel : numpy.ndarray
            The kernel to convolve with shape (H, W, C)
        stride : int
            The stride of the convolution, default is 1
        padding : str or int
            The padding of the convolution, default is 'same'
            - If 'same' the padding is calculated to keep the output size the same as the input size
            - If 'valid' no padding is applied
            - If 'full' the padding is calculated to keep the output size the same as the input size + the kernel size - 1
            - If int, a padding of that size is applied

        Returns
        -------
        numpy.ndarray
            The convolved input, shape (H, W)
        """
        # Get shapes of input and kernel
        if len(input.shape) == 2:
            input = input[:, :, np.newaxis]
        if len(kernel.shape) == 2:
            kernel = kernel[:, :, np.newaxis]
        (iH, iW, iC) = input.shape
        (kH, kW, kC) = kernel.shape

        # Check if the kernel and input have the same number of channels
        if iC != kC:
            raise ValueError(
                "The number of channels in the input and kernel must be the same"
            )

        if padding == "same":
            padding = int((kW - 1) / 2)
        elif padding == "valid":
            padding = 0
        elif padding == "full":
            padding = int(kW - 1)

        # Compute the size of the output volume
        oH = int((iH - kH + 2 * padding) / stride) + 1
        oW = int((iW - kW + 2 * padding) / stride) + 1

        # Initialize the output volume
        output = np.zeros((oH, oW))

        # Pad the input
        input = np.pad(
            input, ((padding, padding), (padding, padding), (0, 0)), "constant"
        )

        # Loop over the output volume
        for y in np.arange(0, oH):
            for x in np.arange(0, oW):
                roi = input[y * stride : y * stride + kH, x * stride : x * stride + kW]

                k = np.sum(roi * kernel)
                output[y, x] = k

        # Return the output volume
        return output

    def _convolve_fast(self, input, kernel, output_shape=None):
        """
        Calculate the convolution of an input with a kernel. Works with any number of channels.

        The function uses `scipy.signal.correlate` to calculate the convolution and hence is faster.

        Parameters
        ----------
        input : numpy.ndarray
            The input to convolve shape (H, W, C)
        kernel : numpy.ndarray
            The kernel to convolve with shape (H, W, C)
        padding : str or int
            The padding of the convolution, default is 'same'
            - If 'same' the padding is calculated to keep the output size the same as the input size
            - If 'valid' no padding is applied
            - If 'full' the padding is calculated to keep the output size the same as the input size + the kernel size - 1
            - If int, a padding of that size is applied

        Returns
        -------
        numpy.ndarray
            The convolved input, shape (H, W)
        """
        ho, wo = output_shape
        output = np.zeros((ho, wo))
        ci = input.shape[-1]

        # Loop over the color channel and use scipy correlate
        for c in range(ci):
            output += corr(input[:, :, c], kernel[:, :, c], mode="valid")

        # Return the output volume
        return output

    def max_pool(self, input, kernel_size, stride=1):
        """
        Calculate the maxpool of an input.

        The function uses `scipy.signal.correlate` to calculate the convolution and hence is faster.

        Parameters
        ----------
        input : numpy.ndarray
            The input to convolve shape (H, W, C)
        kernel_size : int
            The size of the pooling window
        stride : int
            The stride of the pooling, default is 1

        Returns
        -------
        numpy.ndarray
            The convolved input, shape (H, W)
        """
        hi, wi, ci, m = input.shape
        ho = int((hi - kernel_size[0]) / stride) + 1
        wo = int((wi - kernel_size[1]) / stride) + 1

        output = np.zeros((ho, wo, ci, m))
        for i in range(m):
            output[:, :, :, i] = self.max_pool_one(
                input[:, :, :, i], kernel_size, stride
            )
        return output

    def max_pool_one(self, input, kernel_size, stride=1):
        """
        Calculate the maxpool of an input.

        The function uses `scipy.signal.correlate` to calculate the convolution and hence is faster.

        Parameters
        ----------
        input : numpy.ndarray
            The input to convolve shape (H, W, C)
        kernel_size : int
            The size of the pooling window
        stride : int
            The stride of the pooling, default is 1

        Returns
        -------
        numpy.ndarray
            The convolved input, shape (H, W)
        """
        hi, wi, ci = input.shape
        ho = int((hi - kernel_size[0]) / stride) + 1
        wo = int((wi - kernel_size[1]) / stride) + 1
        output = np.zeros((ho, wo, ci))

        # for x in range(wo):
        #     for y in range(ho):
        #         roi = input[
        #             y * stride : y * stride + kernel_size[0],
        #             x * stride : x * stride + kernel_size[1],
        #         ]
        #         # max_id = np.argmax(roi)
        #         output[y, x] = np.max(roi)

        for c in range(ci):
            for x in range(wo):
                for y in range(ho):
                    roi = input[
                        y * stride : y * stride + kernel_size[0],
                        x * stride : x * stride + kernel_size[1],
                        c,
                    ]
                    # max_id = np.argmax(roi)
                    output[y, x, c] = np.max(roi)
        return output

    def average_pool(self, input, kernel_size, stride=1):
        """
        Calculate the averagepool of an input.

        The function uses `scipy.signal.correlate` to calculate the convolution and hence is faster.

        Parameters
        ----------
        input : numpy.ndarray
            The input to convolve shape (H, W, C)
        kernel_size : int
            The size of the pooling window
        stride : int
            The stride of the pooling, default is 1

        Returns
        -------
        numpy.ndarray
            The convolved input, shape (H, W)
        """
        hi, wi, ci = input.shape
        ho = int((hi - kernel_size[0]) / stride) + 1
        wo = int((wi - kernel_size[1]) / stride) + 1
        output = np.zeros((ho, wo, ci))

        for c in range(ci):
            for x in range(wo):
                for y in range(ho):
                    roi = input[
                        y * stride : y * stride + kernel_size[0],
                        x * stride : x * stride + kernel_size[1],
                        c,
                    ]
                    output[y, x, c] = np.mean(roi)
        return output

    def convolve_backward(self, input, kernels, output_grad, stride=1, padding="same"):
        """
        Calculate the backward pass of the convolutional layer.

        The function uses `scipy.signal.correlate` to calculate the convolution and hence is faster.

        Parameters
        ----------
        input : numpy.ndarray
            The input to convolve shape (H, W, C)
        kernels : numpy.ndarray
            The kernels to convolve with shape (H, W, C, K)
        output_grad : numpy.ndarray
            The gradient of the loss with respect to the output of the convolution, shape (H, W, K)
        stride : int
            The stride of the convolution, default is 1
        padding : str or int
            The padding of the convolution, default is 'same'
            - If 'same' the padding is calculated to keep the output size the same as the input size
            - If 'valid' no padding is applied
            - If 'full' the padding is calculated to keep the output size the same as the input size + the kernel size - 1
            - If int, a padding of that size is applied

        Returns
        -------
        numpy.ndarray
            The gradient of the loss with respect to the input, shape (H, W, C)
        """
        # TODO Implement the backward pass for the convolutional layer

        # Calculate the padding
        if padding == "same":
            pad = int((kernels.shape[0] - 1) / 2)
        elif padding == "valid":
            pad = 0
        elif padding == "full":
            pad = kernels.shape[0] - 1

        # Calculate the output shape
        ho = int((input.shape[0] - kernels.shape[0] + 2 * pad) / stride) + 1
        wo = int((input.shape[1] - kernels.shape[1] + 2 * pad) / stride) + 1

        # Calculate the gradient of the loss with respect to the input
        input_grad = np.zeros(input.shape)
        for c in range(kernels.shape[-1]):
            for x in range(wo):
                for y in range(ho):
                    input_grad += corr(
                        output_grad[:, :, c],
                        kernels[:, :, :, c],
                        mode="valid",
                    )

        # Return the gradient of the loss with respect to the input
        return input_grad

    def convolve_backward_one(
        self, input, kernels, output_grad, stride=1, padding="same"
    ):
        # TODO Implement the backward pass for the convolutional layer

        # Calculate the padding
        if padding == "same":
            pad = int((kernels.shape[0] - 1) / 2)
        elif padding == "valid":
            pad = 0
        elif padding == "full":
            pad = kernels.shape[0] - 1

        # Calculate the output shape
        ho = int((input.shape[0] - kernels.shape[0] + 2 * pad) / stride) + 1
        wo = int((input.shape[1] - kernels.shape[1] + 2 * pad) / stride) + 1

        # Calculate the gradient of the loss with respect to the input
        input_grad = np.zeros(input.shape)
        for c in range(kernels.shape[-1]):
            for x in range(wo):
                for y in range(ho):
                    input_grad[
                        y * stride : y * stride + kernels.shape[0],
                        x * stride : x * stride + kernels.shape[1],
                        c,
                    ] += (
                        kernels[:, :, c] * output_grad[y, x, c]
                    )

    def convolve_filter_backward(
        self, input, kernels, output_grad, stride=1, padding="same"
    ):
        # TODO Implement the backward pass for the convolutional layer
        pass

    def max_pool_backward(self, input, pool_size, output_grad, stride=1):
        """
        Calculate the gradient of the output with respect to the input.

        Parameters
        ----------
        input : numpy.ndarray
            The input to the maxpool layer, shape (H, W, C)
        pool_size : int
            The size of the pooling window
        output_grad : numpy.ndarray
            The gradient of the loss with respect to the output of the maxpool layer, shape (H, W, C)
        stride : int
            The stride of the pooling, default is 1

        Returns
        -------
        numpy.ndarray
            The gradient of the loss with respect to the input, shape (H, W, C)
        """
        # TODO: Implement the maxpool backward pass
        pass

    def average_pool_backward(self, input, pool_size, output_grad, stride=1):
        """
        Calculate the gradient of the output with respect to the input.

        Parameters
        ----------
        input : numpy.ndarray
            The input to the averagepool layer, shape (H, W, C)
        pool_size : int
            The size of the pooling window
        output_grad : numpy.ndarray
            The gradient of the loss with respect to the output of the averagepool layer, shape (H, W, C)
        stride : int
            The stride of the pooling, default is 1

        Returns
        -------
        numpy.ndarray
            The gradient of the loss with respect to the input, shape (H, W, C)
        """
        # TODO Implement the averagepool backward
        pass

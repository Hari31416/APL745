# module for managing the data

import numpy as np


def initializers(method, shape, seed=False):
    """
    Initialize the parameters of a layer.

    Parameters
    ----------
    seed : int
        Seed for the random number generator.
    method : str
        Method for initializing the parameters.
    shape : tuple, (n_[l], n_[l-1])
        Shape of the parameters.

    Returns
    -------
    W : ndarray
        Initialized weights.
    """

    if seed:
        np.random.seed(seed)

    if method == 'zeros':
        W = np.zeros(shape)
    if method == 'ones':
        W = np.ones(shape)
    elif method == 'random':
        W = np.random.randn(*shape)
    elif method == 'uniform':
        W = np.random.uniform(-1, 1, shape)
    elif method == 'normal':
        W = np.random.normal(0, 1, shape)
    elif method == 'glorot_uniform':
        limit = np.sqrt(6 / np.sum(shape))
        W = np.random.uniform(-limit, limit, shape)
    elif method == 'glorot_normal':
        std = np.sqrt(2 / np.sum(shape))
        W = np.random.normal(0, std, shape)
    elif method == 'he_uniform':
        limit = np.sqrt(6 / shape[0])
        W = np.random.uniform(-limit, limit, shape)
    elif method == 'he_normal':
        std = np.sqrt(2 / shape[0])
        W = np.random.normal(0, std, shape)
    else:
        raise ValueError('Invalid initialization method')
    
    return W





def get_shapes_cnn(padding, strides, input_shape, kernel_shape):
    """
    Get the output shape of a convolutional layer
    """

    stride_height, stride_width = strides
    input_height, input_width, _ = input_shape
    kernel_height, kernel_width = kernel_shape

    if padding == 'same':
        output_height = int(np.ceil(float(input_height) / float(stride_height)))
        output_width = int(np.ceil(float(input_width) / float(stride_width)))
        padding_height = max((output_height - 1) * stride_height + kernel_height - input_height, 0)
        padding_width = max((output_width - 1) * stride_width + kernel_width - input_width, 0)
        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        # compute the output shape after the padding
        output_height = (input_height + padding_top + padding_bottom - kernel_height) // stride_height + 1
        output_width = (input_width + padding_left + padding_right - kernel_width) // stride_width + 1
    elif padding == 'valid':
        output_height = int(np.ceil(float(input_height - kernel_height + 1) / float(stride_height)))
        output_width = int(np.ceil(float(input_width - kernel_width + 1) / float(stride_width)))
        padding_top = padding_bottom = padding_left = padding_right = 0


    return (output_height, output_width), (padding_top, padding_bottom, padding_left, padding_right)
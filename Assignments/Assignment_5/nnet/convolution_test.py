import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve as conv
from scipy.signal import correlate as corr
from convolution import Convolution


def convolve(image, kernel, stride=1, padding="same"):
    """
    Calculate the convolution of an image with a kernel (This is actually a cross-correlation)

    Parameters
    ----------
    image : numpy.ndarray
        The image to convolve shape (H, W, C)
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
        The convolved image, shape (H, W)
    """
    # Get shapes of image and kernel
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    if len(kernel.shape) == 2:
        kernel = kernel[:, :, np.newaxis]
    (iH, iW, iC) = image.shape
    (kH, kW, kC) = kernel.shape

    # Check if the kernel and image have the same number of channels
    if iC != kC:
        raise ValueError(
            "The number of channels in the image and kernel must be the same"
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

    # Pad the image
    image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), "constant")

    # Loop over the output volume
    for y in np.arange(0, oH):
        for x in np.arange(0, oW):
            roi = image[y * stride : y * stride + kH, x * stride : x * stride + kW]

            k = np.sum(roi * kernel)
            output[y, x] = k

    # Return the output volume
    return output


def convolution_scipy(image, kernel, padding):
    # Get shapes of image and kernel
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    if len(kernel.shape) == 2:
        kernel = kernel[:, :, np.newaxis]
    (iH, iW, iC) = image.shape
    (kH, kW, kC) = kernel.shape

    # Check if the kernel and image have the same number of channels
    if iC != kC:
        raise ValueError(
            "The number of channels in the image and kernel must be the same"
        )

    if padding == "same":
        padding = int((kW - 1) / 2)
    elif padding == "valid":
        padding = 0
    elif padding == "full":
        padding = int(kW - 1)

    # Compute the size of the output volume
    oH = int((iH - kH + 2 * padding)) + 1
    oW = int((iW - kW + 2 * padding)) + 1

    # Initialize the output volume
    output = np.zeros((oH, oW))

    # Pad the image
    image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), "constant")

    # Loop over the color channel and use scipy correlate
    for c in range(iC):
        output += corr(image[:, :, c], kernel[:, :, c], mode="valid")

    # Return the output volume
    return output


con = Convolution()

hi, wi, ci = 128, 128, 1
hk, wk, co = 5, 5, 10
stride = 1
padding = "same"

image = np.random.randint(0, 255, (hi, wi, ci))
kernels = np.random.randint(0, 255, (hk, wk, ci, co))
biases = np.random.randint(0, 255, (co,))
from_class = con.convolve(image, kernels, stride=stride, padding=padding, bias=biases)
print("Shape using class ", from_class.shape)

if isinstance(padding, int):
    padding = padding
elif padding == "same":
    padding = int((hk - 1) / 2)
elif padding == "valid":
    padding = 0
elif padding == "full":
    padding = int(hk - 1)

ho = int((hi - hk + 2 * padding) / stride) + 1
wo = int((wi - wk + 2 * padding) / stride) + 1
output_slow = np.zeros((ho, wo, co))

for i in range(co):
    output_slow[:, :, i] = (
        convolve(image, kernels[:, :, :, i], padding=padding, stride=stride) + biases[i]
    )

print("Slow output shape ", output_slow.shape)

ho_2 = int((hi - hk + 2 * padding)) + 1
wo_2 = int((wi - wk + 2 * padding)) + 1
output_fast = np.zeros((ho_2, wo_2, co))

for i in range(co):
    output_fast[:, :, i] = (
        convolution_scipy(image, kernels[:, :, :, i], padding=padding) + biases[i]
    )

output_fast = output_fast[::stride, ::stride, :]
print("Fast output shape ", output_fast.shape)

try:
    assert (
        np.allclose(output_slow, output_fast)
        == np.allclose(output_slow, from_class)
        == True
    )
    print("All good")
except AssertionError:
    print("Convolutions are not the same")

hi, wi, ci = 128, 128, 3
hk, wk, co = 2, 2, 10
stride = 2
padding = "same"
image = np.random.randint(0, 255, (hi, wi, ci))
ho = int((hi - hk) / stride) + 1
wo = int((wi - wk) / stride) + 1
output_here = np.zeros((ho, wo))

for x in range(wo):
    for y in range(ho):
        roi = image[
            y * stride : y * stride + hk,
            x * stride : x * stride + wk,
        ]
        output_here[y, x] = np.max(roi)

output_class = con.max_pool(image, 2, 2)

assert np.allclose(output_here, output_class) == True, "Max pooling is not the same"

for x in range(wo):
    for y in range(ho):
        roi = image[
            y * stride : y * stride + hk,
            x * stride : x * stride + wk,
        ]
        output_here[y, x] = np.mean(roi)

output_class = con.average_pool(image, 2, 2)

assert np.allclose(output_here, output_class) == True, "Average pooling is not the same"

from scipy.signal import convolve2d
import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from scipy.ndimage import filters


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


def read_image(filename, representation):
    """
    reading the image
    :param filename - path to image:
    :param representation - int:
    :return picture in grayscale or rgb according to the input
    """
    im = imread(filename)
    if representation == 1:  # If the user specified they need grayscale image,
        if len(im.shape) == 3:  # AND the image is not grayscale yet
            im = rgb2gray(im)  # convert to grayscale (**Assuming its RGB and not a different format**)
    im_float = im.astype(np.float64)  # Convert the image type to one we can work with.
    if im_float.max() > 1:  # If image values are out of bound, normalize them.
        im_float = im_float / 255
    return im_float


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    bulids a gaussian pyramid as shown in class
    :param im - matrix represents an image:
    :param max_levels - the maximal number of levels in the resulting pyramid:
    :param filter_size - an odd scalar that represents a squared filter:
    :return pyr - a standard python array where each element of the array is a grayscale image:
            filter-vec - row vector of shape (1, filter_size) used for the pyramid construction:
    """
    pyr = [im]
    filter_vec = gaussian_helper(filter_size)
    for level in range(max_levels-1):
        if im.shape[0] <= 16 or im.shape[1] <= 16:
            return pyr, filter_vec
        filtered_im = filters.convolve(im, filter_vec)
        filtered_im = filters.convolve(filtered_im, filter_vec.T)
        im = filtered_im[::2, ::2]
        pyr.append(im)
    return pyr, filter_vec


def gaussian_helper(filter_size):
    """
    calculates the gaussian filter using convolution
    :param filter_size:
    :return normalised np.array which represents the gaussian filter:
    """
    filter_size = max(2, filter_size)
    base = np.array([1, 1])
    if filter_size == 1:
        base = base/2
        return np.reshape(base, (1, filter_size))
    for i in range(1, filter_size-1):
        base = np.convolve(base, np.array([1, 1]))
    base = base / (2**(filter_size-1))
    return np.reshape(base, (1, filter_size))
############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import higra as hg
import numpy as np


def triangular_filter(image, size):
    """
    Compute a triangular filter on the given 2d image.

    The triangular filter is obtained by convolving the image with the kernel
    [1 2 ... size (size + 1) size ... 2 1] / (size + 1)^2 and its transpose.

    @TODO@ add efficient implementation

    :param image: a 2d array
    :param size: a positive integer
    :return: a 2d array with the same shape as image
    """
    kernel = np.asarray(list(range(size + 1)) + list(range(size, 0, -1)))
    im2 = np.pad(image, size, 'symmetric')
    im2 = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='valid'), axis=0, arr=im2)
    im2 = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='valid'), axis=1, arr=im2)

    return im2


def gradient_orientation(gradient_image, scale=4):
    """
    Estimate gradient orientation.

    Reimplementation of similar function from Piotr Dollar's matlab edge tool box.

    :param gradient_image: 2d image with gradient values
    :param scale: a positive integer (size of the triangular filter)
    :return: 2d image with estimated gradient orientation in [0; pi]
    """

    filtered_gradient = triangular_filter(gradient_image, scale)
    dy, dx = np.gradient(filtered_gradient)
    _, dxx = np.gradient(dx)
    dyy, dxy = np.gradient(dy)
    angle = np.mod(np.arctan2(dyy * np.sign(-dxy), dxx), np.pi)
    return angle

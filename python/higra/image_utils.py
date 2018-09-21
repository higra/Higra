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

    :param image:
    :param size:
    :return:
    """
    kernel = np.asarray(list(range(size + 1)) + list(range(size, 0, -1)))
    im2 = np.pad(image, size, 'symmetric')
    im2 = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='valid'), axis=0, arr=im2)
    im2 = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='valid'), axis=1, arr=im2)

    return im2


def gradient_orientation(gradient_image, scale=4):
    """
    Estimate gradient orientation.

    Reimplementation of similar function from Piotr Dollar's matlab edge tool box
    :param gradient_image: 2d image with gradient values
    :param scale:
    :return: 2d image with estimated gradient orientation in [0; pi]
    """

    filtered_gradient = triangular_filter(gradient_image, scale)
    dy, dx = np.gradient(filtered_gradient)
    _, dxx = np.gradient(dx)
    dyy, dxy = np.gradient(dy)
    angle = np.mod(np.arctan2(dyy * np.sign(-dxy), dxx), np.pi)
    return angle


@hg.data_consumer("shape", "edge_weights")
def mean_pb_hierarchy(graph, shape, edge_weights, edge_orientations=None):
    """
    Compute the mean pb hierarchy as described in :
        P. Arbelaez, M. Maire, C. Fowlkes and J. Malik, "Contour Detection and Hierarchical Image Segmentation,"
        in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 33, no. 5, pp. 898-916, May 2011.
        doi: 10.1109/TPAMI.2010.161

    This does not include gradient estimation.

    :param graph: must be a 4 adjacency graph
    :param shape: (height, width)
    :param edge_weights: gradient value on edges
    :param edge_orientations: estimates orientation of the gradient on edges
    :return: the hierarchy defined on the gradient watershed super-pixels
    """

    shape = hg.__normalize_shape(shape)
    rag, vertex_map, edge_map, tree, altitudes = hg._mean_pb_hierarchy(graph, shape, edge_weights, edge_orientations)


    hg.set_attribute(rag, "vertex_map", vertex_map)
    hg.set_attribute(rag, "edge_map", edge_map)
    hg.set_attribute(rag, "pre_graph", graph)

    hg.set_attribute(tree, "leaf_graph", rag)
    hg.set_attribute(tree, "altitudes", altitudes)

    return tree
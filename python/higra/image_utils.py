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


@hg.argument_helper(hg.CptEdgeWeightedGraph, ("graph", hg.CptGridGraph))
def mean_pb_hierarchy(edge_weights, graph, shape, edge_orientations=None):
    """
    Compute the mean pb hierarchy as described in :
        P. Arbelaez, M. Maire, C. Fowlkes and J. Malik, "Contour Detection and Hierarchical Image Segmentation,"
        in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 33, no. 5, pp. 898-916, May 2011.
        doi: 10.1109/TPAMI.2010.161

    This does not include gradient estimation.
    The final sigmoid scaling of the hierarchy altitude is not performed.

    :param graph: must be a 4 adjacency graph
    :param shape: (height, width)
    :param edge_weights: gradient value on edges
    :param edge_orientations: estimates orientation of the gradient on edges
    :return: the hierarchy defined on the gradient watershed super-pixels
    """

    shape = hg.normalize_shape(shape)
    rag, vertex_map, edge_map, tree, altitudes = hg._mean_pb_hierarchy(graph, shape, edge_weights, edge_orientations)

    hg.set_attribute(rag, "vertex_map", vertex_map)
    hg.set_attribute(rag, "edge_map", edge_map)
    hg.set_attribute(rag, "pre_graph", graph)
    hg.set_attribute(vertex_map, "domain", graph)
    hg.set_attribute(edge_map, "domain", graph)

    hg.set_attribute(tree, "leaf_graph", rag)
    hg.set_attribute(tree, "altitudes", altitudes)

    return tree


@hg.argument_helper(("graph", hg.CptGridGraph))
def multiscale_mean_pb_hierarchy(fine_edge_weights, others_edge_weights, graph, shape, edge_orientations=None):
    """
    Compute the multiscale mean pb hierarchy as described in :
    J. Pont-Tuset, P. Arbeláez, J. Barron, F. Marques, and J. Malik
    Multiscale Combinatorial Grouping for Image Segmentation and Object Proposal Generation
    IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), vol. 39, no. 1, pp. 128 - 140, 2017.
    and in:
    K.K. Maninis, J. Pont-Tuset, P. Arbeláez and L. Van Gool
    Convolutional Oriented Boundaries: From Image Segmentation to High-Level Tasks
    IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), vol. 40, no. 4, pp. 819 - 833, 2018.

    This does not include gradient estimation.
    The final sigmoid scaling of the hierarchy altitude is not performed.

    :param graph: must be a 4 adjacency graph
    :param shape: (height, width)
    :param fine_edge_weights: edge weights of the finest gradient
    :param others_edge_weights: tuple of gradient value on edges
    :param edge_orientations: estimates orientation of the gradient on edges
    :return: the hierarchy defined on the finest gradient watershed super-pixels
    """

    shape = hg.normalize_shape(shape)
    tree_fine = hg.mean_pb_hierarchy(graph, shape, fine_edge_weights, edge_orientations)
    saliency_fine = hg.saliency(tree_fine)
    super_vertex_fine = hg.labelisation_hierarchy_supervertices(tree_fine)

    other_hierarchies = []
    for edge_weights in others_edge_weights:
        tree_coarse = hg.mean_pb_hierarchy(graph, shape, edge_weights, edge_orientations)
        other_hierarchies.append(tree_coarse)

    aligned_saliencies = hg.align_hierarchies(super_vertex_fine, other_hierarchies)

    for saliency in aligned_saliencies:
        saliency_fine += saliency

    saliency_fine *= (1.0 / (1 + len(others_edge_weights)))

    return hg.mean_pb_hierarchy(graph, shape, saliency_fine)

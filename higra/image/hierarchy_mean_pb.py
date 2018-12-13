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


@hg.argument_helper(hg.CptEdgeWeightedGraph, ("graph", hg.CptGridGraph))
def oriented_watershed(edge_weights, graph, shape, edge_orientations=None):
    """
    Compute the oriented watershed as described in :

        P. Arbelaez, M. Maire, C. Fowlkes and J. Malik, "Contour Detection and Hierarchical Image Segmentation,"
        in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 33, no. 5, pp. 898-916, May 2011.
        doi: 10.1109/TPAMI.2010.161

    This does not include gradient estimation.

    :param graph: must be a 4 adjacency graph
    :param shape: (height, width)
    :param edge_weights: gradient value on edges
    :param edge_orientations: estimates orientation of the gradient on edges
    :return: a pair (rag, rag_edge_weights): the region adjacency graph and its estimated edge_weights
    """

    shape = hg.normalize_shape(shape)
    rag, vertex_map, edge_map, rag_edge_weights = hg._oriented_watershed(graph, shape, edge_weights, edge_orientations)

    hg.CptRegionAdjacencyGraph.link(rag, graph, vertex_map, edge_map)
    hg.CptEdgeWeightedGraph.link(rag_edge_weights, rag)

    return rag, rag_edge_weights


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

    hg.CptRegionAdjacencyGraph.link(rag, graph, vertex_map, edge_map)
    hg.CptHierarchy.link(tree, rag)
    hg.CptValuedHierarchy.link(altitudes, tree)

    return tree, altitudes


@hg.argument_helper(hg.CptEdgeWeightedGraph, ("graph", hg.CptGridGraph))
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
    tree_fine, altitudes_fine = hg.mean_pb_hierarchy(fine_edge_weights, graph=graph, shape=shape, edge_orientations=edge_orientations)
    saliency_fine = hg.saliency(altitudes_fine)
    super_vertex_fine = hg.labelisation_hierarchy_supervertices(altitudes_fine)

    other_hierarchies = []
    for edge_weights in others_edge_weights:
        tree_coarse, altitudes_coarse = hg.mean_pb_hierarchy(edge_weights, graph=graph, shape=shape, edge_orientations=edge_orientations)
        other_hierarchies.append(altitudes_coarse)

    aligned_saliencies = hg.align_hierarchies(super_vertex_fine, other_hierarchies)

    for saliency in aligned_saliencies:
        saliency_fine += saliency

    saliency_fine *= (1.0 / (1 + len(others_edge_weights)))

    return hg.mean_pb_hierarchy(saliency_fine, graph=graph, shape=shape)

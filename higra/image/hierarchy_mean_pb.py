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


@hg.argument_helper(hg.CptGridGraph)
def oriented_watershed(graph, edge_weights, shape, edge_orientations=None):
    """
    Creates a region adjacency graph (rag) with the oriented watershed transform.

    The method is described in:

        P. Arbelaez, M. Maire, C. Fowlkes and J. Malik, "Contour Detection and Hierarchical Image Segmentation,"
        in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 33, no. 5, pp. 898-916, May 2011.
        doi: 10.1109/TPAMI.2010.161

    If no edge orientations are provided, then the weight of a rag edge between region i and j is the mean weight of the
    edges linking a vertex of i to a vertex of j.

    This does not include gradient estimation.

    :param graph: must be a 4 adjacency graph (Concept :class:`~higra.CptGridGraph`)
    :param edge_weights: gradient value on edges
    :param shape: shape of the graph, i.e. a pair (height, width) (deduced from :class:`~higra.CptGridGraph`)
    :param edge_orientations: estimated orientation of the gradient on edges (optional)
    :return: a pair (rag, rag_edge_weights): the region adjacency graph (Concept :class:`~higra.CptRegionAdjacencyGraph`) and its estimated edge_weights
    """

    shape = hg.normalize_shape(shape)
    if edge_orientations is not None:
        edge_weights, edge_orientations = hg.cast_to_common_type(edge_weights, edge_orientations)
    rag, vertex_map, edge_map, rag_edge_weights = hg.cpp._oriented_watershed(graph, shape, edge_weights,
                                                                             edge_orientations)

    hg.CptRegionAdjacencyGraph.link(rag, graph, vertex_map, edge_map)

    return rag, rag_edge_weights


@hg.argument_helper(hg.CptGridGraph)
def mean_pb_hierarchy(graph, edge_weights, shape, edge_orientations=None):
    """
    Mean probability boundary hierarchy.

    The method is described in:

        P. Arbelaez, M. Maire, C. Fowlkes and J. Malik, "Contour Detection and Hierarchical Image Segmentation,"
        in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 33, no. 5, pp. 898-916, May 2011.
        doi: 10.1109/TPAMI.2010.161

    This does not include gradient estimation.

    The returned hierarchy is defined on the gradient watershed super-pixels.

    The final sigmoid scaling of the hierarchy altitude is not performed.

    :param graph: must be a 4 adjacency graph (Concept :class:`~higra.CptGridGraph`)
    :param edge_weights: gradient value on edges
    :param shape: shape of the graph, i.e. a pair (height, width) (deduced from :class:`~higra.CptGridGraph`)
    :param edge_orientations: estimated orientation of the gradient on edges (optional)
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """

    shape = hg.normalize_shape(shape)
    if edge_orientations is not None:
        edge_weights, edge_orientations = hg.cast_to_common_type(edge_weights, edge_orientations)
    rag, vertex_map, edge_map, tree, altitudes = hg.cpp._mean_pb_hierarchy(graph, shape, edge_weights,
                                                                           edge_orientations)

    hg.CptRegionAdjacencyGraph.link(rag, graph, vertex_map, edge_map)
    hg.CptHierarchy.link(tree, rag)

    return tree, altitudes


@hg.argument_helper(hg.CptGridGraph)
def multiscale_mean_pb_hierarchy(graph, fine_edge_weights, others_edge_weights, shape, edge_orientations=None):
    """
    Multiscale mean probability boundary hierarchy.

    The method is described in:

        J. Pont-Tuset, P. Arbeláez, J. Barron, F. Marques, and J. Malik
        Multiscale Combinatorial Grouping for Image Segmentation and Object Proposal Generation
        IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), vol. 39, no. 1, pp. 128 - 140, 2017.

    and in:

        K.K. Maninis, J. Pont-Tuset, P. Arbeláez and L. Van Gool
        Convolutional Oriented Boundaries: From Image Segmentation to High-Level Tasks
        IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), vol. 40, no. 4, pp. 819 - 833, 2018.

    This does not include gradient estimation.

    The returned hierarchy is defined on the gradient watershed super-pixels.

    The final sigmoid scaling of the hierarchy altitude is not performed.

    :param graph: must be a 4 adjacency graph (Concept :class:`~higra.CptGridGraph`)
    :param fine_edge_weights: edge weights of the finest gradient
    :param others_edge_weights: tuple of gradient value on edges
    :param shape: shape of the graph, i.e. a pair (height, width) (deduced from :class:`~higra.CptGridGraph`)
    :param edge_orientations: estimated orientation of the gradient on edges (optional)
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """
    shape = hg.normalize_shape(shape)
    tree_fine, altitudes_fine = hg.mean_pb_hierarchy(graph, fine_edge_weights, shape=shape,
                                                     edge_orientations=edge_orientations)
    saliency_fine = hg.saliency(tree_fine, altitudes_fine)
    super_vertex_fine = hg.labelisation_hierarchy_supervertices(tree_fine, altitudes_fine)

    other_hierarchies = []
    for edge_weights in others_edge_weights:
        tree_coarse, altitudes_coarse = hg.mean_pb_hierarchy(graph, edge_weights, shape=shape,
                                                             edge_orientations=edge_orientations)
        other_hierarchies.append((tree_coarse, altitudes_coarse))

    aligned_saliencies = hg.align_hierarchies(graph, super_vertex_fine, other_hierarchies)

    for saliency in aligned_saliencies:
        saliency_fine += saliency

    saliency_fine *= (1.0 / (1 + len(others_edge_weights)))

    return hg.mean_pb_hierarchy(graph, saliency_fine, shape=shape)

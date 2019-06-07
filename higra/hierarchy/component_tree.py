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


def component_tree_min_tree(graph, vertex_weights):
    """
    Min Tree hierarchy from the input vertex weighted graph.

    The Min/Max Tree structure were proposed in [1]_, [2]_.
    The algorithm used in this
    implementation was first described in [3]_.

    .. [1] Ph. Salembier, A. Oliveras, and L. Garrido, "Anti-extensive connected operators for image \
    and sequence processing," IEEE Trans. Image Process., vol. 7, no. 4, pp. 555-570, Apr. 1998.
    .. [2] Ro. Jones, "Connected filtering and segmentation using component trees," Comput. Vis. \
    Image Understand., vol. 75, no. 3, pp. 215-228, Sep. 1999.
    .. [3] Ch. Berger, T. Geraud, R. Levillain, N. Widynski, A. Baillard, and E. Bertin, "Effective \
    Component Tree Computation with Application to Pattern Recognition in Astronomical Imaging," \
    IEEE ICIP 2007.

    :param graph: input graph
    :param vertex_weights: vertex weights of the input graph
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """
    vertex_weights = hg.linearize_vertex_weights(vertex_weights, graph)

    res = hg.cpp._component_tree_min_tree(graph, vertex_weights)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes


def component_tree_max_tree(graph, vertex_weights):
    """
    Max Tree hierarchy from the input vertex weighted graph.

    The Min/Max Tree structure were proposed in [1]_, [2]_.
    The algorithm used in this
    implementation was first described in [3]_.

    :param graph: input graph
    :param vertex_weights: vertex weights of the input graph
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """
    vertex_weights = hg.linearize_vertex_weights(vertex_weights, graph)

    res = hg.cpp._component_tree_max_tree(graph, vertex_weights)
    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes

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


@hg.argument_helper(hg.CptHierarchy)
def accumulate_on_contours(tree, node_weights, accumulator, leaf_graph):
    """
    For each edge of the leaf graph, accumulates the weights of the nodes whose contour pass by this edge.

    For any edge :math:`\{x,y\}`, let :math:`R_{\{x,y\}}` be the set of regions of the input tree :math:`T`
    having :math:`\{x,y\}` in its contour:

    .. math::

        R_{\{x,y\}} = \{n \in T \, |\, |\{x,y\} \cap n| = 1  \}

    The output value for the edge :math:`\{x,y\}` is then the accumulated weights of the nodes
    in :math:`R_{\{x,y\}}`.

    :Runtime complexity:

    This algorithm runs in :math:`\mathcal{O}(n*k)` with :math:`n` the number of edges in the leaf graph and
    :math:`k` the maximal depth of the tree (i.e. the number of edges on the longest downward path between
    the root and a leaf).

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param node_weights: weights on the nodes of the tree
    :param accumulator: see :class:`~higra.Accumulators`
    :param leaf_graph: graph of the tree leaves (deduced from :class:`~higra.CptHierarchy`)
    :return: returns leaf graph edge weights
    """

    depth = hg.attribute_depth(tree)

    res = hg.cpp._accumulate_on_contours(leaf_graph, tree, node_weights, depth, accumulator)
    return res



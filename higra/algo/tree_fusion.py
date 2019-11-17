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


def tree_fusion_depth_map(*trees):
    """
    Depth map associated to the fusion of the given list of trees.
    This depth map can be used to compute a new tree representing the fusion of the input trees, eg. with :func:`~higra.component_tree_max_tree`.

    The method is described in:

        E. Carlinet.
        `A Tree of shapes for multivariate images <https://pastel.archives-ouvertes.fr/tel-01280131/file/TH2015PESC1118.pdf>`_.
        PhD Thesis, Université Paris-Est, 2015.

    All trees must be defined over the same domain, i.e. have the same number of leaves.
    Given a set of trees :math:`(T_1, T_2, T_3,... T_n)` composed of the set of nodes :math:`(N_1, N_2, N_3, ... N_n)`.
    We define the fusion graph as the graph induced the inclusion relation :math:`\subseteq` on the union of all the tree nodes
    :math:`\\bigcup\{N_1, N_2, N_3, ... N_n\}`.

    The result is a directed acyclic graph with a single root (corresponding to the roots of the input trees).
    The depth of a node in this graph is defined as the length of the longest path from the root this node.
    This function returns the depth of the leaves of this graph (which are the same as the leaves of the input trees).

    :See:

    A multivariate tree of shapes of a colour 2d can be computed with
    :func:`~higra.component_tree_multivariate_tree_of_shapes_image2d`.

    :Complexity:

    The worst case runtime complexity of this method is :math:`\mathcal{O}(N^2D^2)` with :math:`N` the number of leaves
    and :math:`D` the number of trees. If we denote by :math:`K` the depth of the deepest tree to merge, one can rewrite
    the runtime complexity as :math:`\mathcal{O}(NKD^2)`.

    :param trees: at least two trees defined over the same domain
    :return: a depth map representing the fusion of the input trees
    """

    return hg.cpp._tree_fusion_depth_map(*trees)

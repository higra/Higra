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


def binary_hierarchy_to_scipy_linkage_matrix(tree, altitudes=None, area=None):
    """
    Converts an Higra binary hierarchy to a SciPy linkage matrix.

    From SciPy documentation:

        An `n-1` by 4 matrix Z is returned.
        At the `i`-th iteration, clusters with indices Z[i, 0] and Z[i, 1] are combined to form cluster `n+i`.
        A cluster with an index  less than `n` corresponds to one of the `n` original observations.
        The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2].
        The fourth value Z[i, 3] represents the number of original observations in the newly formed cluster.

    If `altitudes` is not specified, the value provided by :func:`~higra.attribute_regular_altitudes` on `tree` is used.

    If `area` is not specified, the value provided by :func:`~higra.attribute_area` on `tree` is used.

    :param tree: Input tree
    :param altitudes: Tree nodes altitudes (should be increasing w.r.t tree)
    :param area: Tree nodes area (should be increasing w.r.t tree)
    :return: A linkage matrix
    """

    if altitudes is None:
        altitudes = hg.attribute_regular_altitudes(tree)

    if area is None:
        area = hg.attribute_area(tree)

    area = hg.cast_to_dtype(area, np.int64)
    return hg.cpp._binary_hierarchy_to_scipy_linkage_matrix(tree, altitudes, area)


def scipy_linkage_matrix_to_binary_hierarchy(linkage_matrix):
    """
    Converts a SciPy linkage matrix to an Higra binary hierarchy.

    The result is composed of

        - a tree
        - an array containing the altitudes of the tree nodes
        - an array containing the area of the tree nodes

    :param linkage_matrix: a 2d array as produced by the `linkage` method of SciPy
    :return: a tuple (tree, altitudes, area)
    """
    tree, altitudes, area = hg.cpp._scipy_linkage_matrix_to_binary_hierarchy(linkage_matrix)

    return tree, altitudes, area

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


@hg.extend_class(hg.Tree, method_name="find_region")
def __find_region(self, vertex, level, altitudes):
    """
    Searches for the largest node of altitude lower than the given level and containing the given vertex.
    If no such node exists the given vertex is returned.

    :param vertex: a vertex or a 1d array of vertices
    :param level: a level or a 1d array of levels (should have the same dtype as altitudes)
    :param altitudes: altitudes of the nodes of the tree
    :return: a vertex or a 1d array of vertices
    """

    if isinstance(vertex, np.ndarray):
        if not isinstance(level, np.ndarray):
            level = np.full_like(vertex, level, dtype=altitudes.dtype)
        else:
            level = hg.cast_to_dtype(level, altitudes.dtype)
    else:
        if np.issubdtype(altitudes.dtype, np.integer):
            level = int(level)
        else:
            level = float(level)

    result = self._find_region(vertex, level, altitudes)

    return result


@hg.extend_class(hg.Tree, method_name="child")
def __child(self, index, vertex=None):
    """
    Get the :attr:`index`-th (starting at 0) child of the given vertex/array of vertices.

    If :attr:`vertex` is ``None``, the function will return the :attr:`index`-th child of every non leaf
    node of the tree.

    :param index: positive integer
    :param vertex: a vertex index or a 1d array of vertex indices
        (default to ``np.arange(self.num_leaves(), self.num_vertices()``)
    :return: a vertex index or a 1d array of vertex indices
    """

    index = int(index)

    if vertex is None:
        vertex = np.arange(self.num_leaves(), self.num_vertices())

    result = self._child(index, vertex)

    return result


@hg.extend_class(hg.Tree, method_name="num_children")
def __num_children(self, vertex=None):
    """
    Get the the number of children of the given vertices.

    If :attr:`vertex` is ``None``, the function will return the number of children of every non leaf
    node of the tree.

    :param vertex: a vertex index or a 1d array of vertex indices
        (default to ``np.arange(self.num_leaves(), self.num_vertices()``)
    :return: an integer or a 1d array of integers
    """

    if vertex is None:
        vertex = np.arange(self.num_leaves(), self.num_vertices())

    return self._num_children(vertex)

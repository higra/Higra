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

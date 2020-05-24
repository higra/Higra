############################################################################
# Copyright ESIEE Paris (2020)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import higra as hg
import numpy as np


def tree_monotonic_regression(tree, altitudes, mode, weights=None):
    if mode == "least_square":
        altitudes = hg.cast_to_dtype(altitudes, np.float64)

    if weights is None:
        return hg.cpp._tree_monotonic_regression(tree, altitudes, mode)
    else:
        return hg.cpp._tree_monotonic_regression(tree, altitudes, mode, weights)

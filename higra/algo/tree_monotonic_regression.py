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
    """
    Monotonic regression on the given tree altitudes. Computes new altitudes ``naltitudes`` that are *close* to the given
    :attr:`altitudes` and that are increasing for the given :attr:`tree`: i.e. for any nodes :math:`i, j` such that
    :math:`j` is an ancestor of :math:`i`, then :math:`naltitudes[i] \leq naltitudes[j]`.

    The definition of *close* depends of the value of :attr:`mode`:

    - If :attr:`mode` is equal to ``"min"`` then ``naltitudes`` is the largest increasing function
      below :attr:`altitudes`.
    - If :attr:`mode` is equal to ``"max"`` then ``naltitudes`` is the smallest increasing function
      above :attr:`altitudes`.
    - If :attr:`mode` is equal to ``"least_square"`` then ``naltitudes`` minizes the following minization problem:

    .. math::

        naltitudes = \\arg \min_x \sum_i (weights[i] * (altitudes[i] - x[i])^2)

    such that ``naltitudes`` is increasing for :attr:`tree`.

    :Complexity:

    With :math:`n` the number of nodes in the :attr:`tree`:

    - For the modes ``"min"`` and ``"max"``, the runtime complexity is linear :math:`\mathcal{O}(n)`.
    - For the mode ``"least_square"``, the runtime complexity is linearithmic :math:`\mathcal{O}(n\log(n))` and the
      space complexity is linear  :math:`\mathcal{O}(n)`. The algorithm used is described in:

        P. Pardalos and G. Xue
        `'Algorithms for a Class of Isotonic Regression Problems.' <https://link.springer.com/article/10.1007/PL00009258>`_
        Algorithmica (1999) 23: 211. doi:10.1007/PL00009258

    :param tree: input tree
    :param altitudes: node altitudes of the input tree
    :param mode: the regression mode : ``"min"``, ``"max"``, or ``"least_square"``
    :param weights: node weights of the input tree (default to an array of 1s). This parameter is ignored
                    if :attr:`mode` is not ``"least_square"``.
    :return: a 1d array
    """
    if mode == "least_square":
        altitudes = hg.cast_to_dtype(altitudes, np.float64)

    if weights is None:
        return hg.cpp._tree_monotonic_regression(tree, altitudes, mode)
    else:
        return hg.cpp._tree_monotonic_regression(tree, altitudes, mode, weights)

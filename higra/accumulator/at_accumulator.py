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


def accumulate_at(indices, weights, accumulator):
    """
    Accumulate the given weights located at given indices.

    Let :math:`M = max(indices)`. For all :math:`i \in \{0, \ldots, M\}`

    .. math::

        result[i] = accumulator(\{weights[j, :] \mid indices[j] = i  \})

    If a value :math:`i` of :math:`\{0, \ldots, M\}` does not appear in :attr:`indices`, :math:`result[i]` is the empty
    value of the accumulator (see :class:`~higra.Accumulators`). With ``argmin`` and ``argmax``, :math:`result[i]` is
    the position of the selected weight among the weights :math:`weights[j]` such that :math:`indices[j] = i`, in
    increasing order of :math:`j`.

    :param indices: a 1d array of indices (entries equal to :math:`-1` are ignored; any other negative entry
        raises an exception)
    :param weights: a nd-array of shape :math:`(s_1, \ldots, s_n)` such that :math:`s_1=indices.size`
    :param accumulator: see :class:`~higra.Accumulators`
    :return: a nd-array of size :math:`(M + 1, s_2, \ldots, s_n)`
    """
    indices = hg.cast_to_dtype(indices, np.int64)
    return hg.cpp._accumulate_at(indices, weights, accumulator)

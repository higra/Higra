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


def sort(array, stable=False):
    """
    Sort the given array inplace. A parallel algorithm is used if possible.

    If :attr:`stable` is ``True``, the relative order of equivalent elements is maintained (otherwise the ordering of
    equivalent elements may be arbitrary or even non deterministic).

    :Example:

        >>> a = np.asarray((5, 2, 1, 4, 9))
        >>> hg.sort(a)
        >>> a
        [1  2  4  5  9]

    :param array: input array (1d or 2d)
    :param stable: if ``True``, a stable sort is performed.
    :return: nothing
    """
    if stable:
        return hg.cpp._stable_sort(array)
    else:
        return hg.cpp._sort(array)


def arg_sort(array, stable=False):
    """
    Returns the indices that would sort an array. A parallel algorithm is used if possible.

    If :attr:`stable` is ``True``, the relative order of equivalent elements is maintained (otherwise the ordering of
    equivalent elements may be arbitrary or even non deterministic).

    If the array has 2 dimensions, a lexicographic sort is used.

    :Example:

        >>> a = np.asarray((5, 2, 1, 4, 9))
        >>> hg.arg_sort(a)
        (2 1 3 0 4)

        >>> a = np.asarray(((2, 2, 1, 1, 3),
        >>>                 (2, 2, 2, 1, 0))).T
        >>> hg.arg_sort(a, stable=True)
        (3 2 0 1 4)

    :param array: input array (1d or 2d)
    :param stable: if ``True``, a stable sort is performed.
    :return: A 1d array of indices that would sort the input array
    """
    if stable:
        return hg.cpp._stable_arg_sort(array)
    else:
        return hg.cpp._arg_sort(array)

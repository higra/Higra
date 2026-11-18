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


def _is_sort_dtype(dtype):
    """
    True if elements of the given dtype (ignoring byte order) can be sorted by :func:`sort`.
    """
    kind, size = dtype.kind, dtype.itemsize
    return (kind in "iu" and size in (1, 2, 4, 8)) or (kind == "f" and size in (4, 8))


def sort(array, stable=False):
    """
    Sort the given array inplace. A parallel algorithm is used if possible.

    If :attr:`stable` is ``True``, the relative order of equivalent elements is maintained (otherwise the ordering of
    equivalent elements may be arbitrary or even non deterministic).

    The array can be non-contiguous (for example a slice such as ``a[::2]``); its elements are sorted where they
    are stored. Supported dtypes are signed and unsigned integers and ``float32``/``float64``, in native byte
    order; other inputs (other dtypes, or objects that are not NumPy arrays) raise a ``TypeError``. A read-only
    array raises a ``ValueError``.

    :Example:

        >>> a = np.asarray((5, 2, 1, 4, 9))
        >>> hg.sort(a)
        >>> a
        [1  2  4  5  9]

    :param array: input array (1d)
    :param stable: if ``True``, a stable sort is performed.
    :return: nothing
    """
    sort_function = hg.cpp._stable_sort if stable else hg.cpp._sort

    if not isinstance(array, np.ndarray):
        raise TypeError("hg.sort: the input must be a NumPy array, got %s." % type(array).__name__)

    if not array.dtype.isnative or not _is_sort_dtype(array.dtype):
        raise TypeError("hg.sort: unsupported array dtype %s (supported: signed and unsigned integers, float32 "
                        "and float64, in native byte order)." % array.dtype.str)

    if not array.flags.writeable:
        raise ValueError("hg.sort: the array to sort is read-only.")

    if array.flags.c_contiguous and array.flags.aligned:
        return sort_function(array)

    tmp = np.array(array, order="C")
    sort_function(tmp)
    array[...] = tmp
    return None


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

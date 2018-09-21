############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################


def normalize_shape(shape):
    """
    This function ensure that the given shape will be easily convertible
    in a c++ callback (ie. that it won't interfere badly in pybind11
    overload resolution algorithm)
    :param shape:
    :return:
    """
    return tuple(int(i) for i in shape)


def is_in_bijection(a, b):
    """
    Given two numpy arrays a and b returns true iif
        - a and b have same size
        - there exists a bijective function f such that, for all i a(i) = f(b(i))

    :param a:
    :param b:
    :return:
    """
    aa = a.flatten()
    bb = b.flatten()

    if aa.size != bb.size:
        return False

    equiv1 = {}
    equiv2 = {}

    for i in range(aa.size):
        v1 = aa[i]
        v2 = bb[i]

        if v1 in equiv1:
            if equiv1[v1] != v2:
                return False
        else:
            equiv1[v1] = v2

        if v2 in equiv2:
            if equiv2[v2] != v1:
                return False
        else:
            equiv2[v2] = v1

    return True

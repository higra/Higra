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


def make_lca_fast(tree):
    """
    Deprecated: please use :func:`~higra.Tree.lowest_common_ancestor_preprocess`

    Create an object of type :class:`~higra.LCAFast` for the given tree

    :param tree: input tree
    :return: a LCAFast object
    """
    return tree.lowest_common_ancestor_preprocess()


def __reduce_ctr(*args):
    return hg.LCAFast._make_from_state(*args)


@hg.extend_class(hg.LCAFast, method_name="__reduce__")
def ____reduce__(self):
    return __reduce_ctr, self._get_state(), self.__dict__

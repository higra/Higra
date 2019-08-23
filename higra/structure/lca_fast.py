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


@hg.auto_cache
def make_lca_fast(tree):
    """
    Create an object of type :class:`~higra.LCAFast` for the given tree

    :param tree: input tree
    :return: a LCAFast object
    """
    return hg.LCAFast(tree)

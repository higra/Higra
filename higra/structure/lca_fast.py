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

    :param tree: input tree
    :return:
    """
    return tree.lowest_common_ancestor_preprocess()


def __reduce_ctr_lca_st(*args):
    return hg.LCA_rmq_sparse_table._make_from_state(args)


def __reduce_ctr_lca_stb(*args):
    return hg.LCA_rmq_sparse_table_block._make_from_state(args)


@hg.extend_class(hg.LCA_rmq_sparse_table, method_name="__reduce__")
def ____reduce__(self):
    return __reduce_ctr_lca_st, self._get_state(), self.__dict__


@hg.extend_class(hg.LCA_rmq_sparse_table_block, method_name="__reduce__")
def ____reduce__(self):
    return __reduce_ctr_lca_stb, self._get_state(), self.__dict__


LCAFast = hg.LCA_rmq_sparse_table_block

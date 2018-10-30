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


@hg.data_consumer(altitudes="altitudes")
def reconstruct_leaf_data(tree, deleted_nodes, altitudes):
    """
    Each leaf of the tree takes the altitude of its closest non deleted ancestor.

    :param tree:
    :param deleted_nodes:
    :param altitudes:
    :return:
    """
    reconstruction = hg.propagate_sequential(tree,
                                             altitudes,
                                             deleted_nodes)
    return reconstruction[0:tree.num_leaves(), ...]


@hg.data_consumer(altitudes="altitudes")
def labelisation_horizontal_cut(tree, threshold, altitudes):
    """
    Labelize tree leaves according to an horizontal cut in the tree.

    Two leaves are in the same region (ie. have the same label) if
    the altitude of their lowest common ancestor is strictly greater
    than the specified threshold.

    :param tree:
    :param threshold:
    :param altitudes:
    :return:
    """
    return hg._labelisation_horizontal_cut(tree, float(threshold), altitudes)


@hg.data_consumer(altitudes="altitudes")
def labelisation_hierarchy_supervertices(tree, altitudes, handle_rag=True):
    """
    Labelize the tree leaves into supervertices.

    Two leaves are in the same supervertex if they have a common ancestor of altitude 0.

    If handle_rag is True and the provided has been built on a region adjacency graph, then the labelisation
    corresponding to the rag regions is returned.

    This functions guaranties that the labels are in the range [0, num_supervertices-1].

    :param tree:
    :param altitudes:
    :param handle_rag:
    :return:
    """
    graph = hg.get_attribute(tree, "leaf_graph")
    if graph is not None and hg.get_attribute(graph, "vertex_map") is not None and handle_rag:
        return hg.get_attribute(graph, "vertex_map")

    return hg._labelisation_hierarchy_supervertices(tree, altitudes)

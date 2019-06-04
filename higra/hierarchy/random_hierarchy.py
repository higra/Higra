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


def random_binary_partition_tree(num_leaves, asymmetry_probability):
    """
    Generates a random binary trees with a controlled amount of asymmetry/unbalancedness.

    The tree is grown from the root to the leaves.
    At each step, the algorithm randomly select one of the *growable* leaf node of the current tree.
    Two children are added to the selected node; the number of leaf nodes is hence increased by one.
    Then,

      - with probability :math:`1-asymmetry\_probability`, both new children are marked as *growable*
      - with probability :math:`asymmetry\_probability`, only one of the children is marked as *growable*

    The altitudes of the returned hierarchy are obtained with :func:`~higra.attribute_regular_altitudes`:
    *The regular altitudes is comprised between 0 and 1 and is inversely proportional to the depth of a node*.

    A valid minimal connected graph (a tree) is associated to the leaves of the tree.

    :param num_leaves: expected number of leaves in the generated tree
    :param asymmetry_probability: real value between 0 and 1. At 0 the tree is perfectly unbalanced, at 1 it is
            perfectly balanced (if ``num_leaves`` is  a power of 2)
    :return: a tree (Concept :class:`~higra.CptBinaryHierarchy`) and its node altitudes
    """
    import random
    import math

    assert (0 <= asymmetry_probability <= 1)
    num_leaves = int(num_leaves)
    assert (num_leaves > 0)

    parents = np.zeros((num_leaves * 2 - 1,), dtype=np.int64)

    n = 1
    root = {}
    leaves = []
    leaves.append(root)

    all_nodes = [root]

    i = parents.size - 1
    root["parent"] = i

    while n != 2 * num_leaves - 1:

        ni = random.randint(0, math.floor(asymmetry_probability * (len(leaves) - 1)))
        node = leaves[ni]
        del leaves[ni]

        node["i"] = i
        node["left"] = {"parent": i}
        node["right"] = {"parent": i}
        i -= 1
        all_nodes.append(node["left"])
        all_nodes.append(node["right"])
        n += 2

        if random.random() <= asymmetry_probability:
            if random.random() >= 0.5:
                leaves.append(node["right"])
            else:
                leaves.append(node["left"])
        else:
            leaves.append(node["left"])
            leaves.append(node["right"])

    k = 0
    for node in all_nodes:
        if "i" not in node:
            node["i"] = k
            k += 1
        parents[node["i"]] = node["parent"]

    tree = hg.Tree(parents)

    altitudes = hg.attribute_regular_altitudes(tree)

    def _get_associated_mst(tree, altitudes):
        """
        Create a valid edge mst for the given tree (returns an edge weighted undirected graph)
        """
        nb = tree.num_leaves()
        link_v = np.arange(nb)
        link_v = hg.accumulate_sequential(tree, link_v, hg.Accumulators.first)

        g = hg.UndirectedGraph(nb)
        edge_weights = np.zeros((nb - 1,), np.float32)
        for r in tree.leaves_to_root_iterator(include_leaves=False):
            g.add_edge(link_v[tree.child(0, r)], link_v[tree.child(1, r)])
            edge_weights[r - nb] = altitudes[r]

        return g, edge_weights

    mst, edge_weights = _get_associated_mst(tree, altitudes)

    hg.CptBinaryHierarchy.link(tree, mst)

    return tree, altitudes

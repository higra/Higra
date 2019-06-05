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


def accumulate_parallel(tree, node_weights, accumulator):
    """
    Accumulates values of the children of every node :math:`i` in the :math:`node\_weights` array and puts the result
    in output: :math:`output(i) = accumulator(node\_weights(children(i)))`

    :param tree: input tree
    :param node_weights: Weights on the nodes of the tree
    :param accumulator: see :class:`~higra.Accumulators`
    :return: returns new tree node weights
    """
    res = hg.cpp._accumulate_parallel(tree, node_weights, accumulator)
    return res


@hg.argument_helper(hg.CptHierarchy)
def accumulate_sequential(tree, leaf_data, accumulator, leaf_graph=None):
    """
    Sequential accumulation of node values from the leaves to the root.
    For each leaf node :math:`i`, :math:`output(i) = leaf_data(i)`.
    For each node :math:`i` from the leaves (excluded) to the root, :math:`output(i) = accumulator(output(children(i)))`

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param leaf_data: array of weights on the leaves of the tree
    :param accumulator: see :class:`~higra.Accumulators`
    :param leaf_graph: graph of the tree leaves (optional, deduced from :class:`~higra.CptHierarchy`)
    :return: returns new tree node weights
    """
    if leaf_graph is not None:
        leaf_data = hg.linearize_vertex_weights(leaf_data, leaf_graph)
    res = hg.cpp._accumulate_sequential(tree, leaf_data, accumulator)
    return res


def propagate_sequential(tree, node_weights, condition):
    """
    Sequentially propagates parent values to children:
    for each node :math:`i` from the root to the leaves, if :math:`i` is not the root and if :math:`condition(i)` then
    :math:`output(i) = output(tree.parent(i))` otherwise  :math:`output(i) = node\_weights(i)`.

    :param tree: input tree
    :param node_weights: Weights on the nodes of the tree
    :param condition: Boolean array on the nodes of the tree
    :return: returns new tree node weights
    """
    res = hg.cpp._propagate_sequential(tree, node_weights, condition)
    return res


def propagate_sequential_and_accumulate(tree, node_weights, accumulator):
    """
    Sequentially propagates parent values to children and accumulates with current value:
    for each node :math:`i` from the root to the leaves,
    :math:`output(i) = accumulator(node\_weights(i), output(parent(i)))`.

    :param tree: input tree
    :param node_weights: Weights on the nodes of the tree
    :param accumulator: see :class:`~higra.Accumulators`
    :return: returns new tree node weights
    """
    res = hg.cpp._propagate_sequential_and_accumulate(tree, node_weights, accumulator)
    return res


def propagate_parallel(tree, node_weights, condition=None):
    """
    Propagates parent values to children:
    for each node :math:`i`, if :math:`condition(i)` then
    :math:`output(i) = node\_weights(tree.parent(i))` otherwise  :math:`output(i) = node\_weights(i)`.

    The conditional parallel propagator pseudo-code could be::

        # input: a tree t
        # input: an attribute att on the nodes of t
        # input: a condition cond on the nodes of t

        output = copy(input)

        for each node n of t:
            if(cond(n)):
                output[n] = input[t.parent(n)]

            return output

    :param tree: input tree
    :param node_weights: Weights on the nodes of the tree
    :param condition: Boolean array on the nodes of the tree
    :return: returns new tree node weights
    """

    if condition is not None:
        condition = hg.cast_to_dtype(condition, np.bool)
        res = hg.cpp._propagate_parallel(tree, node_weights, condition)
    else:
        res = hg.cpp._propagate_parallel(tree, node_weights)
    return res


@hg.argument_helper(("tree", hg.CptHierarchy))
def accumulate_and_add_sequential(tree, node_weights, leaf_data, accumulator, leaf_graph=None):
    """
    Accumulates node values from the leaves to the root and add the result with the input array.

    For each leaf node :math:`i`, output(i) = :math:`leaf\_data(i)`.
    For each node :math:`i` from the leaves (excluded) to the root, :math:`output(i) = input(i) + accumulator(output(children(i)))`

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param node_weights: Weights on the nodes of the tree
    :param leaf_data: Weights on the leaves of the tree
    :param accumulator: see :class:`~higra.Accumulators`
    :param leaf_graph: graph of the tree leaves (optional, deduced from :class:`~higra.CptHierarchy`)
    :return: returns new tree node weights
    """
    if leaf_graph is not None:
        leaf_data = hg.linearize_vertex_weights(leaf_data, leaf_graph)

    leaf_data, node_weights = hg.cast_to_common_type(leaf_data, node_weights)
    res = hg.cpp._accumulate_and_add_sequential(tree, node_weights, leaf_data, accumulator)

    return res


@hg.argument_helper(hg.CptHierarchy)
def accumulate_and_multiply_sequential(tree, node_weights, leaf_data, accumulator, leaf_graph=None):
    """
    Accumulates node values from the leaves to the root and multiply the result with the input array.

    For each leaf node :math:`i`, :math:`output(i) = leaf_data(i)`.
    For each node :math:`i` from the leaves (excluded) to the root, :math:`output(i) = input(i) + accumulator(output(children(i)))`

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param node_weights: Weights on the nodes of the tree
    :param leaf_data: Weights on the leaves of the tree
    :param accumulator: see :class:`~higra.Accumulators`
    :param leaf_graph: graph of the tree leaves (optional (optional, deduced from :class:`~higra.CptHierarchy`))
    :return: returns new tree node weights
    """
    if leaf_graph is not None:
        leaf_data = hg.linearize_vertex_weights(leaf_data, leaf_graph)

    leaf_data, node_weights = hg.cast_to_common_type(leaf_data, node_weights)
    res = hg.cpp._accumulate_and_multiply_sequential(tree, node_weights, leaf_data, accumulator)

    return res


@hg.argument_helper(hg.CptHierarchy)
def accumulate_and_min_sequential(tree, node_weights, leaf_data, accumulator, leaf_graph=None):
    """
    Accumulates node values from the leaves to the root and takes the minimum of result and the input array.


    For each leaf node :math:`i`, :math:`output(i) = leaf_data(i)`.
    For each node :math:`i` from the leaves (excluded) to the root, :math:`output(i) = \min(input(i), accumulator(output(children(i)))`

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param node_weights: Weights on the nodes of the tree
    :param leaf_data: Weights on the leaves of the tree
    :param accumulator: see :class:`~higra.Accumulators`
    :param leaf_graph: graph of the tree leaves (optional, deduced from :class:`~higra.CptHierarchy`)
    :return: returns new tree node weights
    """
    if leaf_graph is not None:
        leaf_data = hg.linearize_vertex_weights(leaf_data, leaf_graph)

    leaf_data, node_weights = hg.cast_to_common_type(leaf_data, node_weights)
    res = hg.cpp._accumulate_and_min_sequential(tree, node_weights, leaf_data, accumulator)
    return res


@hg.argument_helper(hg.CptHierarchy)
def accumulate_and_max_sequential(tree, node_weights, leaf_data, accumulator, leaf_graph=None):
    """
    Accumulates node values from the leaves to the root and takes the maximum of result and the input array.


    For each leaf node :math:`i`, :math:`output(i) = leaf_data(i)`.
    For each node :math:`i` from the leaves (excluded) to the root, :math:`output(i) = \max(input(i), accumulator(output(children(i)))`


    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param node_weights: Weights on the nodes of the tree
    :param leaf_data: Weights on the leaves of the tree
    :param accumulator: see :class:`~higra.Accumulators`
    :param leaf_graph: graph of the tree leaves (optional, deduced from :class:`~higra.CptHierarchy`)
    :return: returns new tree node weights
    """
    if leaf_graph is not None:
        leaf_data = hg.linearize_vertex_weights(leaf_data, leaf_graph)

    leaf_data, node_weights = hg.cast_to_common_type(leaf_data, node_weights)
    res = hg.cpp._accumulate_and_max_sequential(tree, node_weights, leaf_data, accumulator)
    return res

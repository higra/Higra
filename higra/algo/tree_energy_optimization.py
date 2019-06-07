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


@hg.argument_helper(hg.CptHierarchy)
def labelisation_optimal_cut_from_energy(tree, energy_attribute, accumulator=hg.Accumulators.sum, leaf_graph=None):
    """
    Labelisation of the input tree leaves corresponding to the optimal cut according to the given energy attribute.

    Given a node :math:`i`, the value :math:`energy(i)` represents the energy fo the partial partition composed of the single region :math:`i`.
    Given a node :math:`i`, the energy of the partial partition composed of the children of :math:`i` is given by :math:`accumulator(energy(children(i)))`
    This function computes the partition (ie. a set of node forming a cut of the tree) that has a minimal energy
    according to the definition above.

    Supported accumulators are `hg.Accumulators.sum`, `hg.Accumulators.min`, and `hg.Accumulators.max`.

    The algorithm used is based on dynamic programming and runs in linear time :math:`\mathcal{O}(n)` w.r.t. to the number of nodes in the tree.

    See:

        Laurent Guigues, Jean Pierre Cocquerez, Hervé Le Men.
        `Scale-sets Image Analysis. International <https://hal.archives-ouvertes.fr/hal-00705364/file/ijcv_scale-setV11.pdf>`_
        Journal of Computer Vision, Springer Verlag, 2006, 68 (3), pp.289-317

    and

        Bangalore Ravi Kiran, Jean Serra.
        `Global-local optimizations by hierarchical cuts and climbing energies. <https://hal.archives-ouvertes.fr/hal-00802978v2/document>`_
        Pattern Recognition Letters, Elsevier, 2014, 47 (1), pp.12-24.


    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param energy_attribute: energy value of each node of the input tree
    :param accumulator: see :class:`~higra.Accumulators`
    :param leaf_graph: leaf graph of the input tree (deduced from :class:`~higra.CptHierarchy`)
    :return: a labelisation of the leaves of the tree
    """
    labels = hg.cpp._labelisation_optimal_cut_from_energy(tree, energy_attribute, accumulator)

    if leaf_graph is not None:
        labels = hg.delinearize_vertex_weights(labels, leaf_graph)

    return labels


def hierarchy_to_optimal_energy_cut_hierarchy(tree, data_fidelity_attribute,
                                              regularization_attribute,
                                              approximation_piecewise_linear_function=10):
    """
    Transforms the given hierarchy into its optimal energy cut hierarchy for the given energy terms.
    In the optimal energy cut hierarchy, any horizontal cut corresponds to an optimal energy cut in the original
    hierarchy.

    Each node :math:`i` of the tree is associated to a data fidelity energy :math:`D(i)` and a regularization energy :math:`R(i)`.
    The algorithm construct a new hierarchy with associated altitudes such that the horizontal cut of level lambda
    is the optimal cut for the energy attribute :math:`D + \lambda * R` of the input tree (see function :func:`~higra.labelisation_optimal_cut_from_energy`).
    In other words, the horizontal cut of level :math:`\lambda` in the result is the cut of the input composed of the nodes :math:`N` such that
    :math:`\sum_{r \in N} D(r) + \lambda * R(r)` is minimal.

    PRECONDITION: the regularization energy :math:`R` must be sub additive: for each node :math:`i`: :math:`R(i) \leq \sum_{c\in Children(i)}R(c)`

    The algorithm runs in linear time :math:`\mathcal{O}(n)`

    See:

        Laurent Guigues, Jean Pierre Cocquerez, Hervé Le Men.
        `Scale-sets Image Analysis. International <https://hal.archives-ouvertes.fr/hal-00705364/file/ijcv_scale-setV11.pdf>`_
        Journal of Computer Vision, Springer Verlag, 2006, 68 (3), pp.289-317

    :param tree: input tree
    :param data_fidelity_attribute: 1d array representing the data fidelity energy of each node of the input tree
    :param regularization_attribute: 1d array representing the regularization energy of each node of the input tree
    :param approximation_piecewise_linear_function: Maximum number of pieces used in the approximated piecewise linear model for the energy function (default 10).
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """
    data_fidelity_attribute, regularization_attribute = hg.cast_to_common_type(data_fidelity_attribute, regularization_attribute)
    res = hg.cpp._hierarchy_to_optimal_energy_cut_hierarchy(tree, data_fidelity_attribute, regularization_attribute,
                                                            int(approximation_piecewise_linear_function))
    new_tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(new_tree, hg.CptHierarchy.get_leaf_graph(tree))

    return new_tree, altitudes


@hg.argument_helper(hg.CptHierarchy)
def hierarchy_to_optimal_MumfordShah_energy_cut_hierarchy(tree,
                                                          vertex_weights,
                                                          leaf_graph,
                                                          approximation_piecewise_linear_function=10):
    """
    Transform the given hierarchy into an optimal energy cut hierarchy using the piecewise constant Mumford-Shah energy
    (see function :func:`~higra.hierarchy_to_optimal_energy_cut_hierarchy`).

    In this context:

        - the data fidelity energy assumes a piecewise constant model in each node and is given by the variance of the vertex values inside the node  (see function :func:`~higra.attribute_gaussian_region_weights_model`) multiplied by its area,
        - the regularity energy is given by the length of the perimeter of the node (see function :func:`~higra.attribute_perimeter_length`).

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param vertex_weights: vertex weights of the leaf graph of the input tree
    :param leaf_graph: leaf graph of the input tree (deduced from :class:`~higra.CptHierarchy`)
    :param approximation_piecewise_linear_function: Maximum number of pieces used in the approximated piecewise linear model for the energy function (default 10).
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """
    area = hg.attribute_area(tree, leaf_graph=leaf_graph)
    _, variance = hg.attribute_gaussian_region_weights_model(tree, vertex_weights, leaf_graph)
    perimeter = hg.attribute_perimeter_length(tree, leaf_graph=leaf_graph)

    if variance.ndim > 1:
        variance = np.trace(variance, axis1=1, axis2=2)

    return hierarchy_to_optimal_energy_cut_hierarchy(tree, variance * area, perimeter,
                                                     int(approximation_piecewise_linear_function))


@hg.argument_helper(hg.CptHierarchy)
def attribute_piecewise_constant_Mumford_Shah_energy(tree, vertex_weights, gamma, leaf_graph):
    """
    Piecewise constant Mumford-Shah energy of each node of the input tree.
    The energy of a node is equal to its data fidelity energy plus gamma times its regularization energy.

    For the piecewise constant Mumford-Shah model:

        - the data fidelity energy assumes a piecewise constant model in each node and is given by the variance of the vertex values inside the node  (see function :func:`~higra.attribute_gaussian_region_weights_model`) multiplied by its area,
        - the regularity energy is given by the length of the perimeter of the node (see function :func:`~higra.attribute_perimeter_length`).

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param vertex_weights: vertex weights of the leaf graph of the input tree
    :param gamma: weighting of the regularization term (should be a positive value)
    :param leaf_graph: leaf graph of the input tree (deduced from :class:`~higra.CptHierarchy`)
    :return: a 1d array measuring the energy of each node the input tree
    """
    area = hg.attribute_area(tree, leaf_graph=leaf_graph)
    _, variance = hg.attribute_gaussian_region_weights_model(tree, vertex_weights, leaf_graph)
    perimeter = hg.attribute_perimeter_length(tree, leaf_graph=leaf_graph)

    if variance.ndim > 1:
        variance = np.trace(variance, axis1=1, axis2=2)

    return variance * area + gamma * perimeter


@hg.argument_helper(("graph", "vertex_area"), ("graph", "vertex_perimeter"), ("graph", "edge_length"))
def binary_partition_tree_MumfordShah_energy(graph,
                                             vertex_values,
                                             vertex_area,
                                             vertex_perimeter,
                                             edge_length,
                                             squared_vertex_values=None):
    """
    Binary partition tree according to the Mumford-Shah energy with a constant piecewise model.

    The distance between two regions is equal to the apparition scale of the merged region.

    See:

        Laurent Guigues, Jean Pierre Cocquerez, Hervé Le Men.
        `Scale-sets Image Analysis. International <https://hal.archives-ouvertes.fr/hal-00705364/file/ijcv_scale-setV11.pdf>`_
        Journal of Computer Vision, Springer Verlag, 2006, 68 (3), pp.289-317


    :param graph: input graph
    :param vertex_values: Sum of values inside each vertex of the input graph (1d array for scalar value or 2d array for
        vectorial values, e.g. RGB pixel values)
    :param vertex_area: area of the vertices of the input graph (provided by :func:`~higra.attribute_vertex_area` on `graph`)
    :param vertex_perimeter: perimeter of the vertices of the input graph (provided by :func:`~higra.attribute_vertex_perimeter` on `graph`)
    :param edge_length: length of the frontier represented by the edges of the input graph
        (provided by :func:`~higra.attribute_edge_length` on `graph`)
    :param squared_vertex_values: Sum of squared values inside each vertex of the input graph (1d array for scalar
        value or 2d array for vectorial values, e.g. RGB pixel values).
        If this argument is not provided, it will default to `vertex_values * vertex_values` which is only correct if a
        vertex contains a single value.
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """
    vertex_values = hg.linearize_vertex_weights(vertex_values, graph)
    vertex_area = hg.linearize_vertex_weights(vertex_area, graph)
    vertex_perimeter = hg.linearize_vertex_weights(vertex_perimeter, graph)

    if squared_vertex_values is None:
        squared_vertex_values = vertex_values * vertex_values

    res = hg.cpp._binary_partition_tree_MumfordShah_energy(
        graph,
        vertex_perimeter,
        vertex_area,
        vertex_values,
        squared_vertex_values,
        edge_length)

    tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(tree, graph)

    return tree, altitudes

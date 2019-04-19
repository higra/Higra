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


@hg.argument_helper(hg.CptHierarchy)
def labelisation_optimal_cut_from_energy(tree, energy_attribute, accumulator=hg.Accumulators.sum, leaf_graph=None):
    """
    Computes the labelisation of the input tree leaves corresponding to the optimal cut according to the given energy attribute.

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
    :return: a labelisation of the leaves of the tree (Concept :class:`~higra.CptVertexLabeledGraph` if `leaf_graph` is not `None`)
    """
    labels = hg.cpp._labelisation_optimal_cut_from_energy(tree, energy_attribute, accumulator)

    if leaf_graph is not None:
        hg.CptVertexLabeledGraph.link(labels, leaf_graph)

    return labels


def hierarchy_to_optimal_energy_cut_hierarchy(tree, data_fidelity_attribute, regularization_attribute):
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
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes (Concept :class:`~higra.CptValuedHierarchy`)
    """
    res = hg.cpp._hierarchy_to_optimal_energy_cut_hierarchy(tree, data_fidelity_attribute, regularization_attribute)
    new_tree = res.tree()
    altitudes = res.altitudes()

    hg.CptHierarchy.link(new_tree, hg.CptHierarchy.get_leaf_graph(tree))
    hg.CptValuedHierarchy.link(altitudes, new_tree)

    return new_tree, altitudes

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


def dendrogram_purity(tree, leaf_labels):
    """
    Weighted average of the purity of each node of the tree with respect to a ground truth
    labelization of the tree leaves.
    
    Let :math:`T` be a tree with leaves :math:`V=\{1, \ldots, n\}`.
    Let :math:`C=\{C_1, \ldots, C_K\}` be a partition of :math:`V` into :math:`k` (label) sets.
    
    The purity of a subset :math:`X` of :math:`V` with respect to class :math:`C_\ell\in C` is the fraction of
    elements of :math:`X` that belongs to class :math:`C_\ell`:
    
    .. math::
    
         pur(X, C_\ell) = \\frac{| X \cap C_\ell |}{| X |}.
    
    The purity of :math:`T` is the defined as:
    
    .. math::
    
         pur(T) = \\frac{1}{Z}\sum_{k=1}^{K}\sum_{x,y\in C_k, x\\neq y} pur(lca_T(x,y), C_k)
    
    with :math:`Z=| \{\{x,y\} \subseteq V \mid x\\neq y, \exists k, \{x,y\}\subseteq C_k\} |`.
    
    :See:
    
         Heller, Katherine A., and Zoubin Ghahramani. "`Bayesian hierarchical clustering <https://www2.stat.duke.edu/~kheller/bhcnew.pdf>`_ ."
         Proc. ICML. ACM, 2005.
    
    :Complexity:
    
    The dendrogram purity is computed in :math:`\mathcal{O}(N\\times K \\times C^2)` with :math:`N` the number of nodes
    in the tree, :math:`K` the number of classes, and :math:`C` the maximal number of children of a node in the tree.

    :param tree: input tree
    :param leaf_labels: a 1d integral array of length `tree.num_leaves()`
    :return:  a score between 0 and 1 (higher is better)
    """
    if leaf_labels.ndim != 1 or leaf_labels.size != tree.num_leaves() or leaf_labels.dtype.kind != 'i':
        raise ValueError("leaf_labels must be a 1d integral array of length `tree.num_leaves()`")

    num_l = tree.num_leaves()
    area = hg.attribute_area(tree)

    max_label = np.max(leaf_labels)
    num_labels = max_label + 1
    label_histo_leaves = np.zeros((num_l, num_labels), dtype=np.float64)
    label_histo_leaves[np.arange(num_l), leaf_labels] = 1

    label_histo = hg.accumulate_sequential(tree, label_histo_leaves, hg.Accumulators.sum)
    class_purity = label_histo / area[:, np.newaxis]

    weights = hg.attribute_children_pair_sum_product(tree, label_histo)
    total = np.sum(class_purity[num_l:, :] * weights[num_l:, :])

    return total / np.sum(weights[num_l:])


@hg.argument_helper(hg.CptHierarchy)
def dasgupta_cost(tree, edge_weights, leaf_graph):
    """
    Dasgupta's cost is an unsupervised measure of the quality of a hierarchical clustering of an edge weighted graph.

    Let :math:`T` be a tree representing a hierarchical clustering of the graph :math:`G=(V, E)`.
    Let :math:`w` be a dissimilarity function on the edges :math:`E` of the graph.

    The Dasgupta's cost is define as:

    .. math::

        dasgupta(T, V, E, w) = \sum_{\{x,y\}\in E} \\frac{area(lca_T(x,y))}{w(\{x,y\})}

    :See:

        S. Dasgupta. "`A cost function for similarity-based hierarchical clustering <https://arxiv.org/pdf/1510.05043.pdf>`_ ."
        In Proc. STOC, pages 118–127, Cambridge, MA, USA, 2016

    :Complexity:

    The runtime complexity is :math:`\mathcal{O}(n\log(n) + m)` with :math:`n` the number of nodes in :math:`T` and
    :math:`m` the number of edges in :math:`E`.

    :param tree: Input tree
    :param edge_weights: Edge weights on the leaf graph (dissimilarities)
    :param leaf_graph: Leaf graph of the input tree (deduced from :class:`~higra.CptHierarchy`)
    :return: a real number
    """
    area = hg.attribute_area(tree, leaf_graph=leaf_graph)

    lcaf = tree.lowest_common_ancestor_preprocess()
    lca = lcaf.lca(leaf_graph)

    return np.sum(area[lca] / edge_weights)


@hg.argument_helper(hg.CptHierarchy)
def tree_sampling_divergence(tree, edge_weights, leaf_graph):
    """
    Tree sampling divergence is an unsupervised measure of the quality of a hierarchical clustering of an
    edge weighted graph.
    It measures how well the given edge weighted graph can be reconstructed from the tree alone.
    It is equal to 0 if and only if the given graph can be fully recovered from the tree.

    It is defined as the Kullback-Leibler divergence between the edge sampling model :math:`p` and the independent
    (null) sampling model :math:`q` of the nodes of a tree (see :func:`~higra.attribute_tree_sampling_probability`).

    The tree sampling divergence on a tree :math:`T` is then

    .. math::

        TSD(T) = \sum_{x \in T} p(x) \log\\frac{p(x)}{q(x)}

    The definition of the tree sampling divergence was proposed in:

        Charpentier, B. & Bonald, T. (2019).  `"Tree Sampling Divergence: An Information-Theoretic Metric for \
        Hierarchical Graph Clustering." <https://hal.telecom-paristech.fr/hal-02144394/document>`_ Proceedings of IJCAI.

    :Complexity:

    The tree sampling divergence is computed in :math:`\mathcal{O}(N (\log(N) + C^2) + M)` with :math:`N` the number of
    nodes in the tree, :math:`M` the number of edges in the leaf graph, and :math:`C` the maximal number of children of
    a node in the tree.

    :param tree: Input tree
    :param edge_weights: Edge weights on the leaf graph (similarities)
    :param leaf_graph: Leaf graph of the input tree (deduced from :class:`~higra.CptHierarchy`)
    :return: a real number
    """

    num_l = tree.num_leaves()
    p = hg.attribute_tree_sampling_probability(tree, leaf_graph, edge_weights, 'edge')[num_l:]
    q = hg.attribute_tree_sampling_probability(tree, leaf_graph, edge_weights, 'null')[num_l:]
    index, = np.where(p)
    return np.sum(p[index] * np.log(p[index] / q[index]))

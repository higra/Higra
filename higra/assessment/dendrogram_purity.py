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
    return hg.cpp._dendrogram_purity(tree, leaf_labels)

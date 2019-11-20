/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#pragma once

#include "../graph.hpp"
#include "../attribute/tree_attribute.hpp"
#include "../accumulator/tree_accumulator.hpp"

namespace hg {

    using namespace xt;
    using namespace xt::placeholders;

    /**
     * Weighted average of the purity of each node of the tree with respect to a ground truth
     * labelization of the tree leaves.
     *
     * Let :math:`T` be a tree with leaves :math:`V=\{1, \ldots, n\}`.
     * Let :math:`C=\{C_1, \ldots, C_K\}` be a partition of :math:`V` into :math:`k` (label) sets.
     *
     * The purity of a subset :math:`X` of :math:`V` with respect to class :math:`C_\ell\in C` is the fraction of
     * elements of :math:`X` that belongs to class :math:`C_\ell`:
     *
     * .. math::
     *
     *      pur(X, C_\ell) = \\frac{| X \cap C_\ell |}{| X |}.
     *
     * The purity of :math:`T` is the defined as:
     *
     * .. math::
     *
     *      pur(T) = \\frac{1}{Z}\sum_{k=1}^{K}\sum_{x,y\in C_k, x\\neq y} pur(lca_T(x,y), C_k)
     *
     * with :math:`Z=| \{\{x,y\} \subseteq V \mid x\\neq y, \exists k, \{x,y\}\subseteq C_k\} |`.
     *
     * :See:
     *
     *      Heller, Katherine A., and Zoubin Ghahramani. "`Bayesian hierarchical clustering <https://www2.stat.duke.edu/~kheller/bhcnew.pdf>`_ ."
     *      Proc. ICML. ACM, 2005.
     *
     * :Complexity:
     *
     * The dendrogram purity is computed in :math:`\mathcal{O}(N\\times K \\times C^2)` with :math:`N` the number of nodes
     * in the tree, :math:`K` the number of classes, and :math:`C` the maximal number of children of a node in the tree.
     *
     * @tparam tree_t
     * @tparam T
     * @param tree input tree
     * @param xleaf_labels must be a 1d array with values in [0, max_label]
     * @return a score between 0 and 1 (higher is better)
     */
    template<typename tree_t, typename T>
    auto dendrogram_purity(const tree_t &tree, const xt::xexpression<T> &xleaf_labels) {
        auto &leaf_labels = xleaf_labels.derived_cast();
        hg_assert_1d_array(leaf_labels);
        hg_assert_leaf_weights(tree, leaf_labels);
        hg_assert_integral_value_type(leaf_labels);

        auto num_l = num_leaves(tree);
        auto area = attribute_area(tree);

        auto max_label = xt::amax(leaf_labels)();
        size_t num_labels = max_label + 1;
        array_2d<double> label_histo_leaves = zeros<double>({num_l, num_labels});
        for (index_t i = 0; i < (index_t) num_l; i++) {
            label_histo_leaves(i, leaf_labels(i)) = 1;
        }

        auto label_histo = accumulate_sequential(tree, label_histo_leaves, accumulator_sum());
        auto class_purity = label_histo / view(area, all(), newaxis());

        auto weights = attribute_children_pair_sum_product(tree, label_histo);
        double Z = sum(weights)();
        double total = sum(view(class_purity, range(num_l, _), all()) *
                           view(weights, range(num_l, _), all()))();

        return total / Z;
    }
};

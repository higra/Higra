/***************************************************************************
* Copyright ESIEE Paris (2020)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "../graph.hpp"
#include "xtensor/xview.hpp"
#include "../accumulator/tree_accumulator.hpp"
#include "../structure/fibonacci_heap.hpp"
#include "../structure/unionfind.hpp"

namespace hg {

    namespace tree_monotonic_regression_internal {

        struct heap_node {
            double value;
            index_t node_index;
        };

        // reverse order to obtain a max_heap
        bool operator<(const heap_node &n1, const heap_node &n2) {
            return n1.value > n2.value;
        }

        using heap_type = fibonacci_heap<heap_node>;
        using heap_element_type = heap_type::value_handle;

        template<typename tree_t, typename T, typename Tw>
        auto tree_monotonic_regression_least_square(const tree_t &tree, const xt::xexpression<T> &xaltitudes,
                                                    const xt::xexpression<Tw> &xweights) {
            auto &altitudes = xaltitudes.derived_cast();
            auto &weights = xweights.derived_cast();
            using value_type = typename T::value_type;

            /*
             * Initialization
             */
            array_1d<double> node_weight = weights;
            array_1d<double> node_block_total_weight = weights;
            array_1d<double> node_value = altitudes;
            array_1d<double> node_block_weighted_sum = node_weight * node_value;

            array_1d<double>::shape_type shape({num_vertices(tree)});
            array_1d<heap_type> node_heap = array_1d<heap_type>::from_shape(shape);
            array_1d<heap_element_type> node_heap_handles = array_1d<heap_element_type>::from_shape(shape);

            for (index_t i: leaves_to_root_iterator(tree, leaves_it::include, root_it::exclude)) {
                node_heap_handles(i) = node_heap(parent(i, tree)).push({node_value(i), i});
            }

            // lazy average
            auto node_average_weight = node_block_weighted_sum / node_block_total_weight;

            index_t num_v = num_vertices(tree);
            union_find uf(num_v); // Block maintenance

            /*
             * Main loop IRT_BIN
             */
            for (index_t i: leaves_to_root_iterator(tree)) {
                // index of the representative tree node for the block containing node i
                index_t ic = uf.find(i);

                // while we have violators among our children, fuse current block with the block of the most important violator
                while (!node_heap(ic).empty() &&
                       node_average_weight(ic) < node_heap(ic).top()->get_value().value) {
                    index_t k = node_heap(ic).top()->get_value().node_index; // index of violator child k
                    node_heap(ic).pop();

                    index_t kc = uf.find(k); // index of the representative tree node for the block containing node k

                    index_t new_ic = uf.link(ic, kc); // merge blocks containing i and k
                    index_t new_ik =
                            (new_ic == ic) ? kc : ic; // index of the node that is not representative

                    // update local variable after merge
                    ic = new_ic;

                    // merge block information
                    node_block_weighted_sum(ic) += node_block_weighted_sum(new_ik);
                    node_block_total_weight(ic) += node_block_total_weight(new_ik);
                    node_heap(ic).merge(node_heap(new_ik));
                }

                // update parent heap to reflect the new weight of the block containing node i
                if (root(tree) != i) {
                    node_heap(parent(i, tree)).update(node_heap_handles(i), {node_average_weight(ic), i});
                }
            }

            // final values computation
            array_nd<value_type> result = array_nd<value_type>::from_shape(shape);
            for (index_t i: leaves_to_root_iterator(tree)) {
                result(i) = (value_type) node_average_weight(uf.find(i));
            }

            return result;
        }
    }

    /**
     * Monotonic regression on the given tree altitudes. Computes new altitudes ``naltitudes`` that are *close* to the given
     * :attr:`altitudes` and that are increasing for the given :attr:`tree`: i.e. for any nodes :math:`i, j` such that
     * :math:`j` is an ancestor of :math:`i`, then :math:`naltitudes[i] \leq naltitudes[j]`.
     *
     * The definition of *close* depends of the value of :attr:`mode`:
     *
     * - If :attr:`mode` is equal to ``"min"`` then ``naltitudes`` is the largest increasing function
     *   below :attr:`altitudes`.
     * - If :attr:`mode` is equal to ``"max"`` then ``naltitudes`` is the smallest increasing function
     *   above :attr:`altitudes`.
     * - If :attr:`mode` is equal to ``"least_square"`` then ``naltitudes`` minizes the following minization problem:
     *
     * .. math::
     *
     *     naltitudes = \\arg \min_x \sum_i (weights[i] * (altitudes[i] - x[i])^2)
     *
     * such that ``naltitudes`` is increasing for :attr:`tree`.
     *
     * :Complexity:
     *
     * With :math:`n` the number of nodes in the :attr:`tree`:
     *
     * - For the modes ``"min"`` and ``"max"``, the runtime complexity is linear :math:`\mathcal{O}(n)`.
     * - For the mode ``"least_square"``, the runtime complexity is linearithmic :math:`\mathcal{O}(n\log(n))` and the
     *   space complexity is linear  :math:`\mathcal{O}(n)`. The algorithm used is described in:
     *
     *     P. Pardalos and G. Xue
     *     `'Algorithms for a Class of Isotonic Regression Problems.' <https://link.springer.com/article/10.1007/PL00009258>`_
     *     Algorithmica (1999) 23: 211. doi:10.1007/PL00009258
     *
     * @tparam tree_t
     * @tparam T
     * @tparam Tw
     * @param tree input tree
     * @param xaltitudes tree node altitudes
     * @param xweights tree node weights
     * @param mode "min", "max", or " least_square"
     * @return
     */
    template<typename tree_t, typename T, typename Tw>
    auto tree_monotonic_regression(const tree_t &tree, const xt::xexpression<T> &xaltitudes,
                                   const xt::xexpression<Tw> &xweights, const std::string &mode) {
        auto &altitudes = xaltitudes.derived_cast();
        auto &weights = xweights.derived_cast();
        hg_assert_node_weights(tree, altitudes);
        hg_assert_1d_array(altitudes);
        using value_type = typename T::value_type;

        bool has_weights = false;

        if (weights.size() != 0) {
            hg_assert_node_weights(tree, weights);
            hg_assert_1d_array(weights);
            has_weights = true;
        }


        if (mode == "max") {
            if (has_weights) {
                HG_LOG_WARNING("The argument 'weights' is ignored with the given mode 'max'");
            }

            auto leaf_altitudes = xt::view(altitudes, xt::range(0, num_leaves(tree)));
            return accumulate_and_combine_sequential(tree,
                                                     altitudes,
                                                     leaf_altitudes,
                                                     accumulator_max(),
                                                     [](const value_type &x1, const value_type &x2) {
                                                         return std::max(x1, x2);
                                                     });
        } else if (mode == "min") {
            if (has_weights) {
                HG_LOG_WARNING("The argument 'weights' is ignored with the given mode 'min'");
            }

            return propagate_sequential_and_accumulate(tree, altitudes, accumulator_min());
        } else if (mode == "least_square") {
            if (has_weights) {
                return tree_monotonic_regression_internal::tree_monotonic_regression_least_square(tree, altitudes,
                                                                                                  weights);
            } else {
                return tree_monotonic_regression_internal::tree_monotonic_regression_least_square(tree,
                                                                                                  altitudes,
                                                                                                  xt::ones<double>(
                                                                                                          {num_vertices(
                                                                                                                  tree)}));
            }

        } else {
            hg_assert(false, "Unknown mode '" + mode + "'.");
        }
    }

    template<typename tree_t, typename T>
    auto tree_monotonic_regression(const tree_t &tree, const xt::xexpression<T> &xaltitudes, const std::string &mode) {
        return tree_monotonic_regression(tree, xaltitudes, array_1d<double>{}, mode);
    }
}

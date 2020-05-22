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
#include "../accumulator/tree_accumulator.hpp"

namespace hg {
    template<typename tree_t, typename T>
    auto tree_monotonic_regression(const tree_t &tree, const xt::xexpression<T> &xaltitudes, const std::string &mode) {
        auto &altitudes = xaltitudes.derived_cast();
        hg_assert_node_weights(tree, altitudes);
        hg_assert_1d_array(altitudes);
        using value_type = typename T::value_type;

        array_1d<T> result = array_1d<T>::from_shape({num_vertices(tree)});

        index_t num_v = num_vertices(tree);


        if (mode == "max") {
            auto leaf_altitudes = xt::view(altitudes, xt::range(0, num_leaves(tree)));
            return accumulate_and_combine_sequential(tree,
                                                     altitudes,
                                                     leaf_altitudes,
                                                     accumulator_max(),
                                                     [](const value_type &x1, const value_type &x2) {
                                                         return std::max(x1, x2);
                                                     });
        } else if (mode == "min") {
            return propagate_sequential_and_accumulate(tree, altitudes, accumulator_min());
        }


    }
}

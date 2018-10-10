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
#include "../accumulator/tree_accumulator.hpp"
#include "../structure/array.hpp"
#include "xtensor/xindex_view.hpp"

namespace hg {

    template<typename tree_t, typename T>
    auto attribute_area(const tree_t &tree, const xt::xexpression<T> &xleaf_area) {
        auto &leaf_area = xleaf_area.derived_cast();
        hg_assert(leaf_area.shape()[0] == num_leaves(tree),
                  "leaf_area size does not match the number of leaves in the tree.");
        return accumulate_sequential(tree, leaf_area, accumulator_sum());
    }

    template<typename tree_t>
    auto attribute_area(const tree_t &tree) {
        return attribute_area(tree, xt::ones<long>({num_leaves(tree)}));
    }

    template<typename T1, typename T2>
    auto attribute_volume(const tree &t, const xt::xexpression<T1> &xnode_area, const xt::xexpression<T2> &xnode_altitude) {
        auto & node_area = xnode_area.derived_cast();
        auto & node_altitude = xnode_altitude.derived_cast();
        hg_assert(node_area.dimension() == 1, "node_area must be a 1d array");
        hg_assert(node_altitude.dimension() == 1, "node_altitude must be a 1d array");
        hg_assert(node_area.size() == num_vertices(t), "node_area size does not match the number of nodes in the tree.");
        hg_assert(node_altitude.size() == num_vertices(t), "node_altitude size does not match the number of nodes in the tree.");

        auto & parent = t.parents();
        array_1d<double> volume = xt::empty<double>({t.num_vertices()});
        for (auto i: leaves_to_root_iterator(t)) {
            volume(i) = std::fabs(node_altitude(i) - node_altitude(parent(i))) * node_area(i);
            for (auto c: t.children(i)) {
                volume(i) += volume(c);
            }
        }
        return volume;
    }
}

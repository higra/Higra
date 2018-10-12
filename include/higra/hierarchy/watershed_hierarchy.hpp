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

#include "higra/graph.hpp"
#include "hierarchy_core.hpp"
#include "../attributes/tree_attributes.hpp"

namespace hg {

    namespace watershed_hierarchy_internal {

        template<typename tree_t, typename T1, typename T2>
        auto correct_attribute_BPT(const tree_t &tree,
                                   const T1 &altitude,
                                   const T2 &attribute) {
            using value_type = typename T2::value_type;
            array_1d<value_type> result = xt::empty_like(attribute);
            for (auto n: leaves_iterator(tree)) {
                result(n) = 0;
            }
            for (auto n: leaves_to_root_iterator(tree, leaves_it::exclude, root_it::exclude)) {
                if (altitude(n) != altitude(parent(n, tree))) {
                    result(n) = attribute(n);
                } else {
                    value_type maxc = std::numeric_limits<value_type>::lowest();
                    for (auto c: children_iterator(n, tree)) {
                        maxc = std::max(maxc, attribute(c));
                    }
                    result(n) = maxc;
                }
            }
            result(root(tree)) = attribute(root(tree));
            return result;
        };
    }

    template<typename graph_t, typename T, typename F>
    auto watershed_hierarchy_by_attribute(
            const graph_t &graph,
            const xt::xexpression<T> &xedge_weights,
            const F &attribute_functor) {
        auto &edge_weights = xedge_weights.derived_cast();
        hg_assert(edge_weights.dimension() == 1, "edge_weights must be a 1d array.");
        hg_assert(edge_weights.size() == num_edges(graph),
                  "edge_weights size does not match the number of edges of the graph.");

        auto bptc = bpt_canonical(graph, edge_weights);
        auto &bpt = std::get<0>(bptc);
        auto &altitude = std::get<1>(bptc);
        auto &mst = std::get<2>(bptc);

        auto bpt_attribute = attribute_functor(bpt, altitude);
        auto corrected_attribute = watershed_hierarchy_internal::correct_attribute_BPT(bpt, altitude, bpt_attribute);
        auto persistence = accumulate_parallel(bpt, corrected_attribute, accumulator_min());
        xt::view(persistence, xt::range(0, num_leaves(bpt))) = 0;

        auto mst_edge_weights = xt::view(persistence, xt::range(num_leaves(bpt), num_vertices(bpt)));

        auto bptc2 = bpt_canonical(mst, mst_edge_weights);
        auto &bpt2 = std::get<0>(bptc2);
        auto &altitude2 = std::get<1>(bptc2);

        auto canonical_tree = simplify_tree(bpt2, [&altitude2, &bpt2](index_t i) {
            return altitude2(i) == altitude2(parent(i, bpt2));
        });
        auto canonical_altitude = xt::eval(xt::index_view(altitude2, canonical_tree.second));

        return std::make_pair(std::move(canonical_tree.first), std::move(canonical_altitude));
    };

    template<typename graph_t, typename T1, typename T2>
    auto watershed_hierarchy_by_area(
            const graph_t &graph,
            const xt::xexpression<T1> &edge_weights,
            const xt::xexpression<T2> &vertex_area) {

        return watershed_hierarchy_by_attribute(
                graph,
                edge_weights,
                [&vertex_area](const tree &t, const array_1d<typename T1::value_type> &altitude) {
                    return attribute_area(t, vertex_area);
                });
    };

    template<typename graph_t, typename T1>
    auto watershed_hierarchy_by_area(
            const graph_t &graph,
            const xt::xexpression<T1> &xedge_weights) {
        return watershed_hierarchy_by_area(graph, xedge_weights, xt::ones<long>({num_vertices(graph)}));
    };

    template<typename graph_t, typename T1, typename T2>
    auto watershed_hierarchy_by_volume(
            const graph_t &graph,
            const xt::xexpression<T1> &edge_weights,
            const xt::xexpression<T2> &vertex_area) {

        return watershed_hierarchy_by_attribute(
                graph,
                edge_weights,
                [&vertex_area](const tree &t, const array_1d<typename T1::value_type> &altitude) {
                    return attribute_volume(t, altitude, vertex_area);
                });
    };

    template<typename graph_t, typename T1>
    auto watershed_hierarchy_by_volume(
            const graph_t &graph,
            const xt::xexpression<T1> &xedge_weights) {
        return watershed_hierarchy_by_volume(graph, xedge_weights, xt::ones<long>({num_vertices(graph)}));
    };

    template<typename graph_t, typename T>
    auto watershed_hierarchy_by_dynamics(
            const graph_t &graph,
            const xt::xexpression<T> &edge_weights) {

        return watershed_hierarchy_by_attribute(
                graph,
                edge_weights,
                [](const tree &t, const array_1d<typename T::value_type> &altitude) {
                    return attribute_dynamics(t, altitude);
                });
    };

}

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

#include "common.hpp"
#include "higra/graph.hpp"
#include "hierarchy_core.hpp"
#include "higra/attribute/tree_attribute.hpp"
#include "higra/algo/graph_core.hpp"

namespace hg {

    namespace watershed_hierarchy_internal {

        template<typename tree_t, typename T1, typename T2>
        auto correct_attribute_BPT(const tree_t &tree,
                                   const T1 &altitude,
                                   const T2 &attribute) {
            using value_type = typename T2::value_type;
            tree.compute_children();
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
                        maxc = (std::max)(maxc, (is_leaf(c, tree)) ? 0 : result(c));
                    }
                    result(n) = maxc;
                }
            }
            result(root(tree)) = attribute(root(tree));
            return result;
        };
    }

    /**
     * Computes a hierarchical watershed for the given regional attribute.
     *
     * The algorithm used is described in:
     *
     *   Laurent Najman, Jean Cousty, Benjamin Perret:
     *   Playing with Kruskal: Algorithms for Morphological Trees in Edge-Weighted Graphs. ISMM 2013: 135-146
     *
     * The regional attribute is specified by an attribute functor, that is a function that
     *   - takes 2 input parameter: a binary partition tree, the altitudes of its nodes;
     *   - returns an 1d array giving the attribute value for each node of the input tree.
     * The computed regional attribute must be scalar, positive and increasing
     * (the attribute value of a node is smaller than or equal to the attribute value of its parent).
     *
     * @tparam graph_t
     * @tparam T
     * @tparam F
     * @param graph: input graph
     * @param xedge_weights: input graph edge weights
     * @param attribute_functor: function that computes the attribute value from a tree and its node altitudes
     * @return a node_weighted_tree
     */
    template<typename graph_t, typename T, typename F>
    auto watershed_hierarchy_by_attribute(
            const graph_t &graph,
            const xt::xexpression<T> &xedge_weights,
            const F &attribute_functor) {
        auto &edge_weights = xedge_weights.derived_cast();
        hg_assert_edge_weights(graph, edge_weights);
        hg_assert_1d_array(edge_weights);

        auto bptc = bpt_canonical(graph, edge_weights);
        auto &bpt = bptc.tree;
        auto &altitude = bptc.altitudes;
        auto &mst_edge_map = bptc.mst_edge_map;
        auto mst = subgraph_spanning(graph, mst_edge_map);

        auto bpt_attribute = attribute_functor(bpt, altitude);
        auto corrected_attribute = watershed_hierarchy_internal::correct_attribute_BPT(bpt, altitude, bpt_attribute);
        auto persistence = accumulate_parallel(bpt, corrected_attribute, accumulator_min());
        xt::view(persistence, xt::range(0, num_leaves(bpt))) = 0;

        auto mst_edge_weights = xt::view(persistence, xt::range(num_leaves(bpt), num_vertices(bpt)));

        return bpt_canonical(mst, mst_edge_weights);
    };

    /**
     * Computes a hierarchical watershed for the given minima ordering.
     *
     * The algorithm used is described in:
     *
     *   Laurent Najman, Jean Cousty, Benjamin Perret:
     *   Playing with Kruskal: Algorithms for Morphological Trees in Edge-Weighted Graphs. ISMM 2013: 135-146
     *
     *
     * The ranking ranking of the minima of the given edge weighted graph (G,w) is given as vertex weights with values in
     * {0..n} with n the number of minima of (G,w). It must satisfy the following pre-conditions:
     *   - each minimum of (G,w) contains at least one non zero vertex,
     *   - all non zero vertices in a minimum have the same weight,
     *   - there is no non zero value vertex outside minima, and
     *   - no two minima contain non zero vertices with the same weight.
     *
     * The altitude associated to each minimum is a non decreasing 1d array of size n + 1 with non negative values.
     * Note that the first entry of the minima altitudes array, ie. the value at index 0, does not represent a minimum and
     * its value should be 0.
     *
     * @tparam graph_t
     * @tparam T1
     * @tparam T2
     * @param graph: input graph
     * @param xedge_weights: input graph edge weights
     * @param xminima_ranks: input graph vertex weights containing the rank of each minima of the input edge weighted graph
     * @return a node_weighted_tree
     */
    template<typename graph_t, typename T1, typename T2>
    auto watershed_hierarchy_by_minima_ordering(
            const graph_t &graph,
            const xt::xexpression<T1> &xedge_weights,
            const xt::xexpression<T2> &xminima_ranks) {
        auto &edge_weights = xedge_weights.derived_cast();
        hg_assert_edge_weights(graph, edge_weights);
        hg_assert_1d_array(edge_weights);
        auto &minima_ranks = xminima_ranks.derived_cast();
        hg_assert_vertex_weights(graph, minima_ranks);
        hg_assert_1d_array(minima_ranks);
        hg_assert_integral_value_type(minima_ranks);

        auto bptc = bpt_canonical(graph, edge_weights);
        auto &bpt = bptc.tree;
        auto &mst_edge_map = bptc.mst_edge_map;
        auto mst = subgraph_spanning(graph, mst_edge_map);

        auto extinction = accumulate_sequential(bpt, minima_ranks, accumulator_max());
        xt::view(extinction, xt::range(0, num_leaves(bpt))) = 0;
        auto persistence = accumulate_parallel(bpt, extinction, accumulator_min());
        xt::view(persistence, xt::range(0, num_leaves(bpt))) = 0;

        auto mst_edge_weights = xt::view(persistence, xt::range(num_leaves(bpt), num_vertices(bpt)));

        return bpt_canonical(mst, mst_edge_weights);
    };

    template<typename graph_t, typename T1, typename T2>
    auto watershed_hierarchy_by_area(
            const graph_t &graph,
            const xt::xexpression<T1> &edge_weights,
            const xt::xexpression<T2> &vertex_area) {

        return watershed_hierarchy_by_attribute(
                graph,
                edge_weights,
                [&vertex_area](const tree &t, const auto &altitude) {
                    return attribute_area(t, vertex_area);
                });
    };
 
    template<typename graph_t, typename T1>
    auto watershed_hierarchy_by_area(
            const graph_t &graph,
            const xt::xexpression<T1> &xedge_weights) {
        return watershed_hierarchy_by_area(graph, xedge_weights, xt::ones<index_t>({num_vertices(graph)}));
    };

    template<typename graph_t, typename T1, typename T2>
    auto watershed_hierarchy_by_volume(
            const graph_t &graph,
            const xt::xexpression<T1> &edge_weights,
            const xt::xexpression<T2> &vertex_area) {

        return watershed_hierarchy_by_attribute(
                graph,
                edge_weights,
                [&vertex_area](const tree &t, const auto &altitude) {
                    return attribute_volume(t, altitude, attribute_area(t, vertex_area));
                });
    };

    template<typename graph_t, typename T1>
    auto watershed_hierarchy_by_volume(
            const graph_t &graph,
            const xt::xexpression<T1> &xedge_weights) {
        return watershed_hierarchy_by_volume(graph, xedge_weights, xt::ones<index_t>({num_vertices(graph)}));
    };

    template<typename graph_t, typename T>
    auto watershed_hierarchy_by_dynamics(
            const graph_t &graph,
            const xt::xexpression<T> &edge_weights) {

        return watershed_hierarchy_by_attribute(
                graph,
                edge_weights,
                [](const tree &t, const auto &altitude) {
                    return attribute_dynamics(t, altitude, true);
                });
    };

}

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
#include "../algo/graph_weights.hpp"
#include "higra/structure/unionfind.hpp"
#include "xtensor/xview.hpp"
#include "higra/sorting.hpp"

namespace hg {

    /**
     * Labelize graph vertices according to the given graph cut.
     * Each edge having a non zero value in the given edge_weights
     * are assumed to be part of the cut.
     *
     * @tparam graph_t
     * @tparam T
     * @tparam label_type
     * @param graph
     * @param edge_weights
     * @return
     */
    template<typename graph_t,
            typename T>
    auto graph_cut_2_labelisation(const graph_t &graph,
                                  const xt::xexpression<T> &xedge_weights) {
        HG_TRACE();
        auto &edge_weights = xedge_weights.derived_cast();
        hg_assert_edge_weights(graph, edge_weights);
        hg_assert_1d_array(edge_weights);

        stackv<index_t> stack;
        array_1d<index_t> labels = xt::empty<index_t>({num_vertices(graph)});
        labels.fill(invalid_index);

        index_t current_label = 0;
        for (auto v: vertex_iterator(graph)) {
            if (labels(v) == invalid_index) {
                current_label++;
                labels(v) = current_label;
                stack.push(v);
                while (!stack.empty()) {
                    auto cv = stack.top();
                    stack.pop();
                    for (auto e: out_edge_iterator(cv, graph)) {
                        if (edge_weights(e) == 0) {
                            auto n = target(e, graph);
                            if (labels(n) == invalid_index) {
                                labels(n) = current_label;
                                stack.push(n);
                            }
                        }
                    }
                }
            }
        }

        return labels;
    };

    /**
     * Determine the graph cut that corresponds to a given labeling
     * of the graph vertices.
     * The result is a weighting of the graph edges where edges with
     * a non zero weight are part of the cut.
     *
     * @tparam graph_t
     * @tparam T
     * @param graph
     * @param xvertex_labels
     * @return
     */
    template<typename graph_t,
            typename T>
    auto labelisation_2_graph_cut(const graph_t &graph,
                                  const xt::xexpression<T> &xvertex_labels) {
        HG_TRACE();
        auto &vertex_labels = xvertex_labels.derived_cast();
        hg_assert_vertex_weights(graph, vertex_labels);
        hg_assert_1d_array(vertex_labels);

        return weight_graph<char>(graph, vertex_labels, weight_functions::L0);
    };


    /**
     * A simple structure to hold the result of minimum_spanning_tree function.
     *
     * The structures holds 2 elements:
     *
     *  - the minimum spanning tree (mst)
     *  - a map (mst_edge_map) that indicates for each edge of the mst, the corresponding edge index in the original graph
     *
     * @tparam mst_t
     */
    template<typename mst_t>
    struct minimum_spanning_tree_result {
        mst_t mst;
        array_1d<index_t> mst_edge_map;
    };

    /**
     * Computes a minimum spanning tree of the given edge weighted graph using Kruskal's algorithm.
     *
     * The returned structure contains two elements:
     *
     *  - the minimum spanning tree (mst)
     *  - a map (mst_edge_map) that indicates for each edge of the mst, the corresponding edge index in the input graph
     *
     * If the input graph is not connected, the result is indeed a minimum spanning forest.
     *
     * @tparam graph_t Input graph type
     * @tparam T Input edge weights type
     * @param graph Input graph
     * @param xedge_weights  Input edge weights
     * @return a mst structure
     */
    template<typename graph_t,
            typename T>
    auto minimum_spanning_tree(const graph_t &graph,
                               const xt::xexpression<T> &xedge_weights) {
        HG_TRACE();
        auto &edge_weights = xedge_weights.derived_cast();
        hg_assert_edge_weights(graph, edge_weights);
        hg_assert_1d_array(edge_weights);

        array_1d<index_t> sorted_edges_indices = xt::arange(num_edges(graph));
        stable_sort(sorted_edges_indices.begin(), sorted_edges_indices.end(),
                         [&edge_weights](index_t i, index_t j) { return edge_weights[i] < edge_weights[j]; });

        auto num_points = num_vertices(graph);

        auto num_edge_mst_max = num_points - 1;
        ugraph mst(num_points);
        array_1d<index_t> mst_edge_map = xt::empty<index_t>({num_edge_mst_max});

        union_find uf(num_points);

        size_t num_edge_found = 0;
        index_t i = 0;

        while (num_edge_found < num_edge_mst_max && i < (index_t) sorted_edges_indices.size()) {
            auto ei = sorted_edges_indices[i];
            auto e = edge_from_index(ei, graph);
            auto c1 = uf.find(source(e, graph));
            auto c2 = uf.find(target(e, graph));
            if (c1 != c2) {
                uf.link(c1, c2);
                mst.add_edge(e);
                mst_edge_map(num_edge_found) = ei;
                num_edge_found++;
            }
            i++;
        }

        if (num_edge_found != num_edge_mst_max) {
            return minimum_spanning_tree_result<ugraph>{
                    std::move(mst),
                    xt::view(mst_edge_map, xt::range(0, num_edge_found))};
        } else{
            return minimum_spanning_tree_result<ugraph>{
                    std::move(mst),
                    std::move(mst_edge_map)};
        }

    };

}

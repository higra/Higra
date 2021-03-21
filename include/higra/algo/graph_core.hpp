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

        array_1d<index_t> sorted_edges_indices = stable_arg_sort(edge_weights);

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
        } else {
            return minimum_spanning_tree_result<ugraph>{
                    std::move(mst),
                    std::move(mst_edge_map)};
        }

    };

    /**
     * Compute a spanning subgraph of the given graph composed of the edges of the input graph indicated in the edge_indices array
     *
     * The edges of the subgraph will be in the order given in edge_indices array.
     *
     * @tparam graph_t
     * @tparam T
     * @param graph input graph
     * @param xedges_indices list of edges of the input graph to include in the subgraph
     * @return a spanning subgraph
     */
    template<typename graph_t, typename T>
    auto subgraph_spanning(const graph_t &graph, const xt::xexpression<T> &xedge_indices) {
        auto &edge_indices = xedge_indices.derived_cast();
        hg_assert_1d_array(edge_indices);
        hg_assert_integral_value_type(edge_indices);

        graph_t subgraph(num_vertices(graph));
        for (index_t ei: edge_indices) {
            auto e = edge_from_index(ei, graph);
            add_edge(source(e, graph), target(e, graph), subgraph);
        }

        return subgraph;
    }

    /**
     * Compute the line graph of an undirected graph.
     *
     * The line graph :math:`LG` of an undirected graph :math:`G` is a graph such that:
     *
     * - each vertex of :math:`LG` represents an edge of :math:`G`: the :math:`i`-th vertex of :math:`LG` corresponds to
     *   the :math:`i`-th edge of :math:`G`; and
     * - two vertices :math:`x` and :math:`y` of :math:`LG` are adjacent if their corresponding edges in :math:`G` share
     *   a common extremity. Formally, if  :math:`x` represents the edge :math:`\{i, j \}` and if :math:`y` represents
     *   the edge :math:`\{k, j \}`, then the edge :math:`\{x, y\}` belongs to :math:`LG` if
     *   :math:`\{i, j \} \\cap \{k, j \} \\neq \emptyset`.
     *
     * The line graph is also known as: the covering graph, the derivative, the edge-to-vertex dual,
     * the conjugate, the representative graph, the edge graph, the interchange graph, the adjoint graph, or the
     * derived graph.
     *
     * @param graph
     * @return
     */
    inline
    ugraph line_graph(const ugraph &graph) {
        ugraph lg(num_edges(graph));
        for (auto v: vertex_iterator(graph)) {
            auto it = graph.out_edges_cbegin(v);
            index_t n_out = out_degree(v, graph);
            for (index_t i = 0; i < n_out; ++i) {
                auto &e1 = edge_from_index(it[i], graph);
                for (index_t j = i + 1; j < n_out; ++j) {
                    auto &e2 = edge_from_index(it[j], graph);
                    // the following test prevents multiple edges from being linked several times
                    if (!(e1.source == e2.source && e1.source < v)) {
                        add_edge(e1.index, e2.index, lg);
                    }
                }
            }
        }
        return lg;
    }


    /**
     * See description of the function above
     *
     * @tparam graph_t
     * @param graph
     * @return
     */
    template<typename graph_t>
    ugraph line_graph(const graph_t &graph) {
        ugraph lg(num_edges(graph));
        for (auto v: vertex_iterator(graph)) {
            for (const auto &e1 : out_edge_iterator(v, graph)) {
                for (const auto &e2 : out_edge_iterator(v, graph)) {
                    // do not proceed the same edge twice
                    if (e1.index < e2.index) {
                        // the following test prevents multiple edges from being linked several times
                        if (!(e1.target == e2.target && e1.target < v)) {
                            add_edge(e1.index, e2.index, lg);
                        }
                    }
                }
            }
        }
        return lg;
    }
}

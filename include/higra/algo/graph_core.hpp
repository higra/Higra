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
#include "xtensor/views/xview.hpp"
#include "higra/sorting.hpp"
#include <optional>
#include <vector>
#include <random>
#include <algorithm>

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

    /**
     * Computes the connected components of an undirected graph and returns a labeling of the vertices.
     * 
     * It is a special use of the function :func:`graph_cut_2_labelisation` where the graph cut is the empty set (no edge in the cut). 
     * 
     * The result is an array of size :math:`|V|` where :math:`V` is the vertex set of the input graph. The value at index
     * :math:`i` is equal to the label of the connected component containing vertex :math:`i`. Labels are in the range
     * :math:`[1, n]` where :math:`n` is the number of connected components in the graph.
     * 
     * Complexity: :math:`\mathcal{O}(|V| + |E|)` with :math:`|V|` and :math:`|E|` respectively the number of vertices and edges in the input graph.
     * 
     * :Example:
     * 
     * >>> graph = hg.UndirectedGraph(6)
     * >>> graph.add_edges((0, 1, 3), (1, 2, 4))
     * >>> hg.connected_components_labeling(graph)
     * array([1, 1, 1, 2, 2, 3])
     * 
     * @param graph: input graph
     * @return: an array of size :math:`|V|` containing the connected component labels of each vertex
     */
    template<typename graph_t>
    auto connected_components_labeling(const graph_t &graph) {
        return graph_cut_2_labelisation(graph, xt::zeros<char>({num_edges(graph)}));
    }

    /**
     * Creates a random undirected graph with a given number of vertices and a given mean number of edges per vertex. 
     * The function uses the model of random graphs proposed by Erdős and Rényi. The resulting graph is a simple graph (no self-loops, no multiple edges).
     * 
     * First, a graph is generated with :math:`n` vertices.
     * Then, we compute the probability :math:`p` of having an edge between two vertices with:
     * 
     * .. math::
     * 
     *        p = \frac{d_{mean}}{n-1}
     * 
     * Afterwards, for each pair of vertices, an edge is created with probability :math:`p`.
     * 
     * By default, the generated graph is not guaranteed to be connected. If :attr:`allow_non_connected` is ``False``, the function will add edges to ensure that the graph is connected.
     * 
     * To do so, the function computes the connected components of the graph and linearly connects consecutive components 
     * (by adding an edge between a randomly chosen vertex of component :math:`i-1` and component :math:`i`, for :math:`i \in [1, k-1]` where :math:`k` is the number of connected components).
     * 
     * Complexity: :math:`\mathcal{O}(n^2)` with :math:`n` the number of vertices in the graph.
     * 
     * :Example:
     * >>> graph_no n_connected = hg.random_undirected_graph_erdos_renyi(100, 1.8, True, seed=42)
     * >>> hg.connected_components_labeling(graph_non_connected)
     * array([ 1,  1,  2,  1,  1,  3,  1,  4,  1,  1,  3,  1,  5,  3,  1,  3,  6,
     *     1,  7,  1,  1,  1,  3,  1,  1,  1,  1,  4,  7,  3,  1,  1,  1,  4,
     *     8,  1,  1,  1,  1,  4,  1,  1,  1,  1,  1,  3,  9,  1,  1,  1, 10,
     *     1, 11, 12,  6,  1,  1, 13,  1,  1,  1, 14, 12,  1,  1,  1,  1,  1,
     *    13,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 15,  3, 15,  1,
     *     7,  1,  1,  1,  1,  1,  4, 16,  1,  1,  1,  1,  1,  3,  1])
     * >>> graph_connected = hg.random_undirected_graph_erdos_renyi(100, 1.8, False, seed=42)
     * >>> hg.connected_components_labeling(graph_connected)
     * array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     *    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     *    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     *    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     *    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

     * @param num_vertices number of vertices in the graph
     * @param mean_degree mean number of edges per vertex in the graph
     * @param allow_non_connected if true, the generated graph may be non-connected. Otherwise, the generated graph is guaranteed to be connected.
     * @param seed seed for the random number generator (optional). If not provided, the random number generator will be seeded with the current time.
     * @return The generated undirected graph
     */
    inline
    auto random_undirected_graph_erdos_renyi(const int num_vertices, const double mean_degree, const bool allow_non_connected = true, const std::optional<unsigned int> seed = std::nullopt) {
        hg_assert(num_vertices > 0, "Number of vertices must be positive.");
        hg_assert(mean_degree >= 0, "Mean number of edges per vertex must be non-negative.");
        hg_assert(num_vertices == 1 || mean_degree <= num_vertices-1, "Mean number of edges per vertex must be less than or equal to the maximum number of edges in a simple undirected graph.");

        ugraph g(num_vertices);

        double p = (num_vertices > 1) ? (mean_degree / (num_vertices-1)) : 0.0;

        std::mt19937 gen(seed.has_value() ? seed.value() : std::random_device{}());
        std::bernoulli_distribution dist(p);

        for (index_t i = 0; i < num_vertices; ++i) {
            for (index_t j = i + 1; j < num_vertices; ++j) {
                if (dist(gen)) {
                    g.add_edge(i, j);
                }
            }
        }

        if (!allow_non_connected) {
            array_1d<index_t> labels = connected_components_labeling(g);
            size_t num_components = *std::max_element(labels.begin(), labels.end());
            std::vector<std::vector<index_t>> components(num_components);
            for (index_t i = 0; i < num_vertices; ++i) {
                components[labels(i)-1].push_back(i);
            }
            for (index_t i = 1; i < num_components; ++i) {
                std::uniform_int_distribution<size_t> dist_prev(0, components[i - 1].size() - 1);
                std::uniform_int_distribution<size_t> dist_curr(0, components[i].size() - 1);
                g.add_edge(components[i-1][dist_prev(gen)], components[i][dist_curr(gen)]);
            }
        }

        return g;
    }
}

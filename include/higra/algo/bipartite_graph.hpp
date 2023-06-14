/***************************************************************************
* Copyright ESIEE Paris (2023)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#pragma once

#include "details/csa.hpp"
#include "../structure/unionfind.hpp"
#include "../graph.hpp"

namespace hg {


    /**
     * @brief Check if a graph is bipartite
     *
     * A bipartite graph is a graph whose vertices can be divided into two disjoint and independent sets X and Y such that
     * every edge connects a vertex in X to one in Y.
     *
     * This function is implemented using a depth first search. Its runtime complexity is O(|V| + |E|).
     *
     * If the graph is bipartite, the function returns a pair of a boolean set to true and an array of colors.
     * For any vertex v, color[v] == 0 if v belongs to X and color[v] == 1 if v belongs to Y.
     * Note that the coloring is not unique, the algorithm returns any valid coloring.
     *
     * If the graph is not bipartite, the function returns a pair of a boolean set to false and an empty array.
     *
     * @tparam graph_t type of the input graph
     * @param g input graph
     * @return a pair of a boolean and an array of colors.
     * The boolean is true if the graph is bipartite and false otherwise.
     * If the graph is bipartite, the array contains the color of each vertex.
     * If the graph is not bipartite, the array is empty.
     */
    template<typename graph_t>
    std::pair<bool, array_1d<unsigned char>> is_bipartite_graph(const graph_t &g) {
        array_1d<unsigned char> color({num_vertices(g)}, 2);
        std::stack<index_t> stack;
        for (auto o: vertex_iterator(g)) {
            if (color[o] == 2) {
                stack.push(o);
                color[o] = 0;
                while (!stack.empty()) {
                    auto v = stack.top();
                    stack.pop();
                    for (const auto &n: adjacent_vertex_iterator(v, g)) {
                        if (color[n] == 2) {
                            color[n] = 1 - color[v];
                            stack.push(n);
                        } else if (color[n] == color[v]) {
                            return {false, {}};
                        }
                    }
                }
            }
        }
        return {true, color};
    }


    /**
     * @brief Check if a graph is bipartite
     *
     * A bipartite graph is a graph whose vertices can be divided into two disjoint and independent sets X and Y such that
     * every edge connects a vertex in X to one in Y.
     *
     * This function is implemented using a union find approach. Its runtime complexity is O(|E| α(|V|)) time,
     * where α is the extremely slowly growing inverse of the single-valued Ackermann function.
     *
     * If the graph is bipartite, the function returns a pair of a boolean set to true and an array of colors.
     * For any vertex v, color[v] == 0 if v belongs to X and color[v] == 1 if v belongs to Y.
     * Note that the coloring is not unique, the algorithm returns any valid coloring.
     *
     * If the graph is not bipartite, the function returns a pair of a boolean set to false and an empty array.
     *
     * @tparam T type of the input arrays
     * @param xsources source vertices of the edges
     * @param xtargets target vertices of the edges
     * @param num_vertices number of vertices in the graph
     * @return a pair of a boolean and an array of colors.
     * The boolean is true if the graph is bipartite and false otherwise.
     * If the graph is bipartite, the array contains the color of each vertex.
     * If the graph is not bipartite, the array is empty.
     */
    template<typename T>
    std::pair<bool, array_1d<unsigned char>> is_bipartite_graph(const xt::xexpression<T> &xsources,
                                                         const xt::xexpression<T> &xtargets,
                                                         index_t num_vertices) {
        auto &sources = xsources.derived_cast();
        auto &targets = xtargets.derived_cast();
        hg_assert_1d_array(sources);
        hg_assert_1d_array(targets);
        hg_assert(xt::has_shape(sources, targets.shape()), "sources and targets must have the same shape");
        hg_assert_integral_value_type(sources);
        hg_assert_integral_value_type(targets);

        union_find uf(num_vertices);
        array_1d<index_t> map({(size_t) num_vertices}, invalid_index);
        array_1d<unsigned char> color({(size_t) num_vertices}, 0);

        for (index_t i = 0; i < (index_t)sources.size(); ++i) {
            auto s = sources[i];
            auto t = targets[i];
            auto cs = uf.find(s);
            auto ct = uf.find(t);
            if (cs == ct) {
                return {false, {}};
            }
            if (map[s] == invalid_index) {
                map[s] = ct;
            } else {
                ct = uf.link(uf.find(map[s]), ct);
            }
            if (map[t] == invalid_index) {
                map[t] = cs;
            } else {
                cs = uf.link(uf.find(map[t]), cs);
            }
            color[cs] = 0;
            color[ct] = 1;
        }

        for (index_t i = 0; i < num_vertices; ++i) {
            color[i] = color[uf.find(i)];
        }

        return {true, color};

    }

    /**
     * @brief Bipartite graph matching
     *
     * The bipartite graph matching problem is to find a maximum cardinality matching with minimum weight in a bipartite graph.
     *
     * The input graph must be a balanced bipartite graph, i.e. the number of vertices in each side must be equal.
     * Moreover, the left hand side must be composed of the vertices 0 to num_vertices(graph)/2 - 1 and
     * the right hand side of the vertices num_vertices(graph)/2 to num_vertices(graph) - 1.
     * A perfect match must exist in the graph or this function might never terminate or return an incorrect result.
     *
     * Edge weights must be of integral type.
     *
     * This function is implemented using the CSA library by Andrew V. Goldberg and Robert Kennedy.
     * It is based on a push-relabel method.
     *
     * @tparam graph_t type of the input graph
     * @tparam T type of the weights
     * @param graph input graph
     * @param xedge_weights edge weights
     * @return the indices of the edges in the matching
     */
    template<typename graph_t, typename T>
    auto bipartite_graph_matching(const graph_t &graph, const xt::xexpression<T> &xedge_weights) {
        auto &weights = xedge_weights.derived_cast();
        hg_assert_edge_weights(graph, weights);
        hg_assert_1d_array(weights);
        hg_assert_integral_value_type(weights);
        hg_assert(num_vertices(graph) % 2 == 0, "The number of vertices must be even.");
        hg::graph_algorithms::CSA csa(sources(graph), targets(graph), num_vertices(graph), weights);
        return csa.edge_indices();
    }

}
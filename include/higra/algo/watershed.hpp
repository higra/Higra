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
#include "../structure/array.hpp"
#include "higra/structure/unionfind.hpp"
#include "higra/sorting.hpp"
#include <vector>
#include <stack>

namespace hg {

    /**
     * Linear time watershed cut algorithm.
     *
     * Jean Cousty, Gilles Bertrand, Laurent Najman, Michel Couprie. Watershed Cuts: Minimum Spanning
     * Forests and the Drop of Water Principle. IEEE Transactions on Pattern Analysis and Machine
     * Intelligence, Institute of Electrical and Electronics Engineers, 2009, 31 (8), pp.1362-1374.
     * @tparam graph_t
     * @tparam T
     * @param graph
     * @param xedge_weights
     * @return array of labels on graph vertices, numbered from 1 to n with n the number of minima
     */
    template<typename graph_t, typename T>
    auto
    labelisation_watershed(const graph_t &graph, const xt::xexpression<T> &xedge_weights) {
        HG_TRACE();
        auto &edge_weights = xedge_weights.derived_cast();
        hg_assert_edge_weights(graph, edge_weights);
        hg_assert_1d_array(edge_weights);

        using value_type = typename T::value_type;
        using vertex_t = typename graph_traits<graph_t>::vertex_descriptor;

        auto fminus = array_1d<value_type>::from_shape({graph.num_vertices()});

        for (auto v: vertex_iterator(graph)) {
            auto minValue = (std::numeric_limits<value_type>::max)();
            for (auto e: out_edge_iterator(v, graph)) {
                minValue = (std::min)(minValue, edge_weights(e));
            }
            fminus[v] = minValue;
        }


        auto no_label = (std::numeric_limits<index_t>::max)();
        auto labels = array_1d<index_t>::from_shape({graph.num_vertices()});
        std::fill(labels.begin(), labels.end(), no_label);

        auto notInL = array_1d<bool>::from_shape({graph.num_vertices()});
        std::fill(notInL.begin(), notInL.end(), true);

        std::vector<vertex_t> L;
        std::vector<vertex_t> LL;

        auto stream = [&L, &LL, &graph, &edge_weights, &fminus, &notInL, &labels, no_label](vertex_t x) {
            L.clear();
            LL.clear();
            L.push_back(x);
            LL.push_back(x);
            notInL[x] = false;

            while (!LL.empty()) {
                auto y = LL[LL.size() - 1];
                LL.pop_back();

                for (auto e: out_edge_iterator(y, graph)) {
                    auto adjacent_vertex = target(e, graph);
                    if (notInL[adjacent_vertex] && edge_weights(e) == fminus[y]) {
                        if (labels[adjacent_vertex] != no_label) {
                            return labels[adjacent_vertex];
                        } else if (fminus[adjacent_vertex] < fminus[y]) {
                            L.push_back(adjacent_vertex);
                            notInL[adjacent_vertex] = false;
                            LL.clear();
                            LL.push_back(adjacent_vertex);
                            break; // stop breadth_first
                        } else {
                            L.push_back(adjacent_vertex);
                            notInL[adjacent_vertex] = false;
                            LL.push_back(adjacent_vertex);
                        }
                    }
                }
            }
            return no_label;
        };

        index_t num_labs = 0;

        for (auto v: vertex_iterator(graph)) {
            if (labels[v] == no_label) {
                auto res = stream(v);
                if (res == no_label) {
                    num_labs++;
                    for (auto x: L) {
                        labels[x] = num_labs;
                        notInL[x] = true;
                    }
                } else {
                    for (auto x: L) {
                        labels[x] = res;
                        notInL[x] = true;
                    }
                }
            }
        }
        return labels;
    };


    template<typename graph_t, typename T1, typename T2>
    auto labelisation_seeded_watershed(
            const graph_t &graph,
            const xt::xexpression<T1> &xedge_weights,
            const xt::xexpression<T2> &xvertex_seeds,
            const typename T2::value_type background_label = 0) {
        HG_TRACE();
        auto &edge_weights = xedge_weights.derived_cast();
        auto &vertex_seeds = xvertex_seeds.derived_cast();
        hg_assert_edge_weights(graph, edge_weights);
        hg_assert_node_weights(graph, vertex_seeds);
        hg_assert_1d_array(edge_weights);
        hg_assert_1d_array(vertex_seeds);

        using label_type = typename T2::value_type;

        array_1d<index_t> sorted_edges_indices = stable_arg_sort(edge_weights);

        index_t num_nodes = num_vertices(graph);
        index_t num_edges = sorted_edges_indices.size();

        union_find uf(num_nodes);

        array_1d<label_type> labels = vertex_seeds;

        for (index_t i = 0; i < num_edges; i++) {
            auto ei = sorted_edges_indices[i];
            auto e = edge_from_index(ei, graph);
            auto c1 = uf.find(source(e, graph));
            auto c2 = uf.find(target(e, graph));

            if (c1 != c2 && (labels(c1) == background_label || labels(c2) == background_label)) {
                if (labels(c1) == background_label) {
                    labels(c1) = labels(c2);
                } else {
                    labels(c2) = labels(c1);
                }
                uf.link(c1, c2);
            }

        }

        for (index_t i = 0; i < num_nodes; i++) {
            if (labels(i) == background_label) {
                labels(i) = labels(uf.find(i));
            }
        }

        return labels;
    };

}

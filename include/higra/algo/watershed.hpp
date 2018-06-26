//
// Created by perretb on 16/05/18.
//

#pragma once


#include "../graph.hpp"
#include "../structure/array.hpp"
#include <vector>
#include <stack>

namespace hg {

    /**
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
        hg_assert(edge_weights.dimension() == 1, "Only scalar edge waits are supported");
        hg_assert(edge_weights.size() == graph.num_edges(),
                  "Edge weights size does not match the number of edges in the graph.");

        using value_type = typename T::value_type;
        using vertex_t = typename graph_traits<graph_t>::vertex_descriptor;

        auto fminus = array_1d<value_type>::from_shape({graph.num_vertices()});

        for (auto v: vertex_iterator(graph)) {
            auto minValue = std::numeric_limits<value_type>::max();
            for (auto ei: out_edge_index_iterator(v, graph)) {
                minValue = std::min(minValue, edge_weights[ei]);
            }
            fminus[v] = minValue;
        }


        auto no_label = std::numeric_limits<std::size_t>::max();
        auto labels = array_1d<std::size_t>::from_shape({graph.num_vertices()});
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

                for (auto ei: out_edge_index_iterator(y, graph)) {
                    auto e = edge(ei, graph);
                    auto adjacent_vertex = (source(e, graph) == y) ? target(e, graph) : source(e, graph);
                    if (notInL[adjacent_vertex] && edge_weights[ei] == fminus[y]) {
                        if (labels[adjacent_vertex] != no_label) {
                            return std::make_pair(std::move(L), labels[adjacent_vertex]);
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
            return std::make_pair(std::move(L), no_label);
        };

        std::size_t num_labs = 0;

        for (auto v: vertex_iterator(graph)) {
            if (labels[v] == no_label) {
                auto res = stream(v);
                if (res.second == no_label) {
                    num_labs++;
                    for (auto x: res.first) {
                        labels[x] = num_labs;
                        notInL[x] = true;
                    }
                } else {
                    for (auto x: res.first) {
                        labels[x] = res.second;
                        notInL[x] = true;
                    }
                }
            }
        }
        return labels;
    };


}

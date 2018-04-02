//
// Created by user on 3/9/18.
//

#pragma once

#include "debug.hpp"
#include "utils.hpp"
#include <boost/graph/graph_traits.hpp>
#include "undirected_graph.hpp"
#include "regular_graph.hpp"
#include "tree_graph.hpp"
#include "graph_utils.hpp"

namespace hg{

    template<typename T>
    auto copy_graph(const T &graph) {
        static_assert(
                std::is_base_of<boost::adjacency_graph_tag, typename boost::graph_traits<T>::traversal_category>::value,
                "Graph must implement adjacency graph concept.");
        static_assert(
                std::is_base_of<boost::vertex_list_graph_tag, typename boost::graph_traits<T>::traversal_category>::value,
                "Graph must implement vertex list graph concept.");

        ugraph g(num_vertices(graph));
        for (const auto v : vertex_iterator(graph)) {
            for (const auto n : adjacent_vertex_iterator(v, graph)) {
                if (n > v)
                    g.add_edge(v, n);
            }
        }
        return g;
    };

    template<>
    inline
    auto copy_graph(const ugraph &graph) {
        ugraph g(num_vertices(graph));
        for (const auto e : edge_iterator(graph)) {
            g.add_edge(source(e, graph), target(e, graph));
        }
        return g;
    };


    inline
    regular_grid_graph get_4_adjacency_graph(const embedding_grid &embedding) {
        hg_assert(embedding.dimension() == 2, "4 adjacency graph requires a 2d embedding");
        std::vector<xt::xarray<long>> neighbours{{-1, 0},
                                                 {0,  -1},
                                                 {0,  1},
                                                 {1,  0}}; // 4 adjacency

        return regular_grid_graph(embedding, std::move(neighbours));
    }

    inline
    regular_grid_graph get_8_adjacency_graph(const embedding_grid &embedding) {
        hg_assert(embedding.dimension() == 2, "4 adjacency graph requires a 2d embedding");
        std::vector<xt::xarray<long>> neighbours{{-1, -1},
                                                 {-1, 0},
                                                 {-1, 1},
                                                 {0,  -1},
                                                 {0,  1},
                                                 {1,  -1},
                                                 {1,  0},
                                                 {1,  1}}; // 4 adjacency

        return regular_grid_graph(embedding, std::move(neighbours));
    }
}
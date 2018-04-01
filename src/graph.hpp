//
// Created by user on 3/9/18.
//

#pragma once

#include "debug.hpp"
#include "undirected_graph.hpp"
#include "regular_graph.hpp"
#include "tree_graph.hpp"
#include "graph_utils.hpp"

namespace hg{

    using boost::source;
    using boost::target;
    using boost::out_edges;
    using boost::in_edges;
    using boost::in_degree;
    using boost::out_degree;
    using boost::degree;
    using boost::vertices;
    using boost::edges;
    using boost::add_vertex;
    using boost::add_edge;
    using boost::num_vertices;
    using boost::num_edges;
    using boost::adjacent_vertices;


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
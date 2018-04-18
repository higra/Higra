//
// Created by user on 4/18/18.
//

#pragma once

#include "../graph.hpp"

namespace hg {
    inline
    auto get_4_adjacency_implicit_graph(const embedding_grid_2d &embedding) {
        std::vector<point_2d_i> neighbours{{-1, 0},
                                           {0,  -1},
                                           {0,  1},
                                           {1,  0}}; // 4 adjacency

        return regular_grid_graph_2d(embedding, std::move(neighbours));
    }

    inline
    auto get_8_adjacency_implicit_graph(const embedding_grid_2d &embedding) {
        std::vector<point_2d_i> neighbours{{-1, -1},
                                           {-1, 0},
                                           {-1, 1},
                                           {0,  -1},
                                           {0,  1},
                                           {1,  -1},
                                           {1,  0},
                                           {1,  1}}; // 8 adjacency

        return regular_grid_graph_2d(embedding, std::move(neighbours));
    }

    inline
    auto get_4_adjacency_graph(const embedding_grid_2d &embedding) {
        return copy_graph(get_4_adjacency_implicit_graph(embedding));
    }

    inline
    auto get_8_adjacency_graph(const embedding_grid_2d &embedding) {
        return copy_graph(get_8_adjacency_implicit_graph(embedding));
    }


    template<typename graph_t, typename T, typename result_type = typename T::value_type>
    auto
    contour2d_2_khalimsky(const graph_t &graph, const embedding_grid_2d &embedding, const xt::xexpression<T> &xweight) {
        const auto &weight = xweight.derived_cast();
        hg_assert(weight.dimension() == 1, "Only scalar weights are supported!");

        auto &shape = embedding.shape();

        auto shapew = weight.shape();

        std::vector<long> res_shape = {shape[0] * 2 - 1, shape[1] * 2 - 1};
        for (std::size_t i = 1; i < shapew.size(); ++i)
            res_shape.push_back(shapew[i]);

        xt::xarray<result_type> res = xt::zeros<result_type>(res_shape);

        for (auto ei: edge_index_iterator(graph)) {
            auto e = edge(ei, graph);
            auto s = source(e, graph);
            auto t = target(e, graph);
            if (t > s) {
                auto ti = embedding.lin2grid(t);
                auto si = embedding.lin2grid(s);
                res[ti + si] = weight(ei);
            }
        }

        embedding_grid_2d res_embedding(shape * 2 - 1);
        auto adj4 = get_4_adjacency_implicit_graph(res_embedding);
        auto h = res_embedding.shape()[0];
        auto w = res_embedding.shape()[1];
        auto flat_res = xt::flatten(res);

        for (long y = 1; y < h - 1; y += 2) {
            for (long x = 1; x < w - 1; x += 2) {
                auto v = res_embedding.grid2lin({y, x});
                result_type max_v = std::numeric_limits<result_type>::lowest();
                for (auto av: adjacent_vertex_iterator(v, adj4)) {
                    max_v = std::max(max_v, flat_res(av));
                }
                res(y, x) = max_v;
            }
        }

        return res;
    };
}
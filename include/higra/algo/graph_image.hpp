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
    contour2d_2_khalimsky(const graph_t &graph, const embedding_grid_2d &embedding, const xt::xexpression<T> &xweight,
                          bool add_extra_border = false) {
        const auto &weight = xweight.derived_cast();
        hg_assert(weight.dimension() == 1, "Only scalar weights are supported!");

        auto &shape = embedding.shape();

        auto shapew = weight.shape();

        long border = (add_extra_border) ? 1 : -1;

        std::vector<long> res_shape{shape[0] * 2 + border, shape[1] * 2 + border};
        for (std::size_t i = 1; i < shapew.size(); ++i)
            res_shape.push_back(shapew[i]);

        xt::xarray<result_type> res = xt::zeros<result_type>(res_shape);
        point_2d_i one = {1, 1};
        for (auto ei: edge_index_iterator(graph)) {
            auto e = edge(ei, graph);
            auto s = source(e, graph);
            auto t = target(e, graph);
            if (t > s) {
                auto ti = embedding.lin2grid(t);
                auto si = embedding.lin2grid(s);
                if (add_extra_border)
                    res[ti + si + one] = weight(ei);
                else
                    res[ti + si] = weight(ei);
            }
        }

        embedding_grid_2d res_embedding(res_shape);
        auto adj4 = get_4_adjacency_implicit_graph(res_embedding);
        auto h = res_embedding.shape()[0];
        auto w = res_embedding.shape()[1];
        auto flat_res = xt::flatten(res);

        long ymin = (add_extra_border) ? 0 : 1;
        long ymax = (add_extra_border) ? h : h - 1;
        long xmin = (add_extra_border) ? 0 : 1;
        long xmax = (add_extra_border) ? w : w - 1;

        for (long y = ymin; y < ymax; y += 2) {
            for (long x = xmin; x < xmax; x += 2) {
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
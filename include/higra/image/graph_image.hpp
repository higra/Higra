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
#include <stack>

namespace hg {

    /**
     * Create a 4 adjacency implicit regular graph for the given embedding
     * @param embedding
     * @return
     */
    inline
    auto get_4_adjacency_implicit_graph(const embedding_grid_2d &embedding) {
        std::vector<point_2d_i> neighbours{{{-1, 0}},
                                           {{0,  -1}},
                                           {{0,  1}},
                                           {{1,  0}}}; // 4 adjacency

        return regular_grid_graph_2d(embedding, std::move(neighbours));
    }

    /**
     * Create of 4 adjacency implicit regular graph for the given embedding
     * @param embedding
     * @return
     */
    inline
    auto get_8_adjacency_implicit_graph(const embedding_grid_2d &embedding) {
        std::vector<point_2d_i> neighbours{{{-1, -1}},
                                           {{-1, 0}},
                                           {{-1, 1}},
                                           {{0,  -1}},
                                           {{0,  1}},
                                           {{1,  -1}},
                                           {{1,  0}},
                                           {{1,  1}}}; // 8 adjacency

        return regular_grid_graph_2d(embedding, std::move(neighbours));
    }

    /**
     * Create of 4 adjacency explicit regular graph for the given embedding
     * @param embedding
     * @return
     */
    inline
    auto get_4_adjacency_graph(const embedding_grid_2d &embedding) {
        return hg::copy_graph<ugraph>(get_4_adjacency_implicit_graph(embedding));
    }

    /**
     * Create of 8 adjacency explicit regular graph for the given embedding
     * @param embedding
     * @return
     */
    inline
    auto get_8_adjacency_graph(const embedding_grid_2d &embedding) {
        return hg::copy_graph<ugraph>(get_8_adjacency_implicit_graph(embedding));
    }


    /**
     * Represents a 4 adjacency edge weighted regular graph in 2d Khalimsky space
     * @param embedding
     * @return
     */
    template<typename graph_t, typename T, typename result_type = typename T::value_type>
    auto
    graph_4_adjacency_2_khalimsky(const graph_t &graph, const embedding_grid_2d &embedding,
                          const xt::xexpression<T> &xedge_weights,
                          bool add_extra_border = false,
                          result_type extra_border_value = 0) {
        HG_TRACE();
        const auto &weight = xedge_weights.derived_cast();
        hg_assert_edge_weights(graph, weight);
        hg_assert_1d_array(weight);
        hg_assert(num_vertices(graph) == embedding.size(),
                  "Graph number of vertices does not match the size of the embedding.");
        auto &shape = embedding.shape();

        index_t border = (add_extra_border) ? 1 : -1;

        std::array<index_t, 2> res_shape{(index_t)shape[0] * 2 + border, (index_t)shape[1] * 2 + border};

        array_2d <result_type> res = xt::zeros<result_type>(res_shape);

        point_2d_i one{{1, 1}};
        for (auto e: edge_iterator(graph)) {
            auto s = source(e, graph);
            auto t = target(e, graph);
            if (t > s) {
                auto ti = embedding.lin2grid(t);
                auto si = embedding.lin2grid(s);
                if (add_extra_border)
                    res[ti + si + one] = weight(e);
                else
                    res[ti + si] = weight(e);
            }
        }

        embedding_grid_2d res_embedding(res_shape);
        auto adj4 = get_4_adjacency_implicit_graph(res_embedding);
        auto h = res_embedding.shape()[0];
        auto w = res_embedding.shape()[1];
        auto flat_res = xt::flatten(res);

        if (add_extra_border && extra_border_value != 0) {
            for (index_t x = 1; x < w; x += 2) {
                res(0, x) = extra_border_value;
                res(h - 1, x) = extra_border_value;
            }
            for (index_t y = 1; y < h; y += 2) {
                res(y, 0) = extra_border_value;
                res(y, w - 1) = extra_border_value;
            }
        }

        index_t ymin = (add_extra_border) ? 0 : 1;
        index_t ymax = (add_extra_border) ? h : h - 1;
        index_t xmin = (add_extra_border) ? 0 : 1;
        index_t xmax = (add_extra_border) ? w : w - 1;

        for (index_t y = ymin; y < ymax; y += 2) {
            for (index_t x = xmin; x < xmax; x += 2) {
                auto v = res_embedding.grid2lin({y, x});
                result_type max_v = std::numeric_limits<result_type>::lowest();
                for (auto av: adjacent_vertex_iterator(v, adj4)) {
                    max_v = (std::max)(max_v, flat_res(av));
                }
                res(y, x) = max_v;
            }
        }

        return res;
    };

    /**
     * Transforms a contour map represented in 2d Khalimsky space into a weighted 4 adjacency edge weighted regular graph
     * (0-face and 2-face of the Khalimsky space are ignored).
     * @param embedding
     * @return
     */
    template<typename T, typename result_type = typename T::value_type>
    auto
    khalimsky_2_graph_4_adjacency(const xt::xexpression<T> &xkhalimsky, bool extra_border = false) {
        HG_TRACE();
        const auto &khalimsky = xkhalimsky.derived_cast();
        hg_assert(khalimsky.dimension() == 2, "Only 2d khalimsky grids are supported!");

        auto &shape = khalimsky.shape();

        index_t border = (extra_border) ? 0 : 1;

        std::array<index_t, 2> res_shape{(index_t) shape[0] / 2 + border, (index_t) shape[1] / 2 + border};
        embedding_grid_2d res_embedding(res_shape);

        auto g = get_4_adjacency_graph(res_embedding);
        array_1d <result_type> weights = xt::zeros<result_type>({num_edges(g)});

        point_2d_i one{{1, 1}};
        for (auto e : edge_iterator(g)) {
            auto s = res_embedding.lin2grid(source(e, g));
            auto t = res_embedding.lin2grid(target(e, g));
            if (extra_border) {
                weights(e) = khalimsky[s + t + one];
            } else {
                weights(e) = khalimsky[s + t];
            }
        }

        return std::make_tuple(std::move(g), std::move(res_embedding), std::move(weights));
    };

}
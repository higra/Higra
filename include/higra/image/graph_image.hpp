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
#include <unordered_map>

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

        std::array<index_t, 2> res_shape{(index_t) shape[0] * 2 + border, (index_t) shape[1] * 2 + border};

        array_2d<result_type> res = xt::zeros<result_type>(res_shape);

        point_2d_i one{{1, 1}};
        for (const auto &e: edge_iterator(graph)) {
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
     * @param xkhalimsky 2d array representing the khalimsky space
     * @param g 4 adjacency graph, with the same number of vertices as the embedding
     * @param embedding embedding of the graph, dimension is half the size of the khalimsky space
     * @param extra_border if true, the khalimsky space will have an extra border of 0-faces and 1-faces
     * @return 2d array representing the weights of the edges in the khalimsky space
     */
    template<typename T, typename result_type = typename T::value_type>
    auto
    khalimsky_2_graph_4_adjacency(const xt::xexpression<T> &xkhalimsky, const ugraph &g,
                                  const embedding_grid_2d &embedding, bool extra_border = false) {
        HG_TRACE();
        const auto &khalimsky = xkhalimsky.derived_cast();
        hg_assert(khalimsky.dimension() == 2, "Only 2d khalimsky grids are supported!");

        auto &shape = khalimsky.shape();

        index_t border = (extra_border) ? 0 : 1;

        hg_assert(num_vertices(g) == embedding.size(), "Graph size does not match.");
        hg_assert(embedding.shape()[0] == (index_t) shape[0] / 2 + border, "Embedding shape[0] does not match.");
        hg_assert(embedding.shape()[1] == (index_t) shape[1] / 2 + border, "Embedding shape[1] does not match.");

        array_1d<result_type> weights = xt::zeros<result_type>({num_edges(g)});

        point_2d_i one{{1, 1}};
        for (const auto &e: edge_iterator(g)) {
            auto s = embedding.lin2grid(source(e, g));
            auto t = embedding.lin2grid(target(e, g));
            if (extra_border) {
                weights(e) = khalimsky[s + t + one];
            } else {
                weights(e) = khalimsky[s + t];
            }
        }

        return weights;
    };

    /**
     * Transforms a contour map represented in 2d Khalimsky space into a weighted 4 adjacency edge weighted regular graph
     * (0-face and 2-face of the Khalimsky space are ignored).
     * @param xkhalimsky 2d array representing the khalimsky space
     * @param extra_border if true, the khalimsky space will have an extra border of 0-faces and 1-faces
     * @return A tuple composed of the 4 adjacency graph, the embedding of the graph, and the weights of the edges in the khalimsky space
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
        array_1d<result_type> weights = khalimsky_2_graph_4_adjacency(khalimsky, g, res_embedding, extra_border);

        return std::make_tuple(std::move(g), std::move(res_embedding), std::move(weights));
    };


    /**
     * Creates a bipartite graph, linking each pixel of the first contour image to any pixel of the second contour image
     * that is within a given distance.
     *
     * @tparam T
     * @param embedding
     * @param xcontour_image1
     * @param xcontour_image2
     * @param max_distance
     * @return
     */
    template<typename T>
    auto get_bipartite_matching_graph_contour_image_2d(const embedding_grid_2d &embedding,
                                                       const xt::xexpression<T> &xcontour_image1,
                                                       const xt::xexpression<T> &xcontour_image2,
                                                       double max_distance) {


        HG_TRACE();
        auto &contour_image1 = xcontour_image1.derived_cast();
        auto &contour_image2 = xcontour_image2.derived_cast();
        hg_assert_same_shape(contour_image1, contour_image2);
        hg_assert(contour_image1.dimension() == 2, "Only 2d contour images are supported!");
        hg_assert(embedding.shape()[0] == (index_t)contour_image1.shape()[0], "Embedding shape[0] does not match.");
        hg_assert(embedding.shape()[1] == (index_t)contour_image1.shape()[1], "Embedding shape[1] does not match.");

        index_t max_distance_sup = (index_t)std::ceil(max_distance);
        hg_assert(max_distance >= 0, "Max distance must be positive.");

        struct neighbor {
            index_t x;
            index_t y;
            double distance;
        };

        /* We precompute the neighbors of each pixel */
        std::vector<neighbor> neighbors;
        neighbors.reserve((size_t) std::ceil(4 * max_distance_sup * max_distance_sup));
        for (index_t y = -max_distance_sup; y <= max_distance_sup; ++y) {
            for (index_t x = -max_distance_sup; x <= max_distance_sup; ++x) {
                double distance = std::sqrt(x * x + y * y);
                if (distance <= max_distance) {
                    neighbors.push_back({x, y, distance});
                }
            }
        }

        std::vector<index_t> node_map1;
        std::unordered_map<index_t, index_t> node_map2;

        index_t height = embedding.shape()[0];
        index_t width = embedding.shape()[1];
        std::vector<index_t> sources, targets;
        std::vector<double> weights;
        for (index_t y = 0; y < height; ++y) {
            for (index_t x = 0; x < width; ++x) {
                if (contour_image1(y, x) != 0) {
                    bool found = false;
                    for (auto &n: neighbors) {
                        index_t x2 = x + n.x;
                        index_t y2 = y + n.y;
                        if (x2 >= 0 && x2 < width && y2 >= 0 && y2 < height) {
                            if (contour_image2(y2, x2) != 0) {
                                index_t lin2 = embedding.grid2lin({y2, x2});
                                if (!found) {
                                    index_t lin1 = embedding.grid2lin({y, x});
                                    node_map1.push_back(lin1);
                                    found = true;
                                }
                                index_t num_node1 = node_map1.size() - 1;
                                auto it = node_map2.find(lin2);
                                if (node_map2.count(lin2) == 0) {
                                    auto nelem = node_map2.size();
                                    node_map2[lin2] = nelem;
                                }
                                index_t num_node2 = node_map2[lin2];
                                sources.push_back(num_node1);
                                targets.push_back(num_node2);
                                weights.push_back(n.distance);
                            }
                        }
                    }
                }
            }
        }

        size_t num_nodes1 = node_map1.size();
        size_t num_nodes2 = node_map2.size();

        array_1d<index_t> node_map = array_1d<index_t>::from_shape({num_nodes1 + num_nodes2});

        for (size_t i = 0; i < num_nodes1; ++i) {
            node_map[i] = node_map1[i];
        }

        for (auto &p: node_map2) {
            node_map[p.second + num_nodes1] = p.first;
        }

        array_1d<index_t> sources_finalized = xt::adapt(sources, {sources.size()});
        array_1d<index_t> targets_finalized = xt::adapt(targets, {targets.size()}) + num_nodes1;
        array_1d<double> weights_finalized = xt::adapt(weights, {weights.size()});

        return std::make_tuple(std::move(sources_finalized), std::move(targets_finalized), std::move(weights_finalized),
                               std::move(node_map), num_nodes1, num_nodes2);
    }
}
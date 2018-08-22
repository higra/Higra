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
    contour2d_2_khalimsky(const graph_t &graph, const embedding_grid_2d &embedding,
                          const xt::xexpression<T> &xedge_weights,
                          bool add_extra_border = false,
                          result_type extra_border_value = 0) {
        HG_TRACE();
        const auto &weight = xedge_weights.derived_cast();
        hg_assert(weight.dimension() == 1, "Edge weights must be scalar.");
        hg_assert(num_edges(graph) == weight.size(),
                  "Edge weights size does not match the number of edge in the graph.");
        hg_assert(num_vertices(graph) == embedding.size(),
                  "Graph number of vertices does not match the size of the embedding.");
        auto &shape = embedding.shape();

        long border = (add_extra_border) ? 1 : -1;

        std::array<long, 2> res_shape{shape[0] * 2 + border, shape[1] * 2 + border};

        array_2d <result_type> res = xt::zeros<result_type>(res_shape);

        point_2d_i one{{1, 1}};
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

        if (add_extra_border && extra_border_value != 0) {
            for (long x = 1; x < w; x += 2) {
                res(0, x) = extra_border_value;
                res(h - 1, x) = extra_border_value;
            }
            for (long y = 1; y < h; y += 2) {
                res(y, 0) = extra_border_value;
                res(y, w - 1) = extra_border_value;
            }
        }

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

    /**
     * Transforms a contour map represented in 2d Khalimsky space into a weighted 4 adjacency edge weighted regular graph
     * (0-face and 2-face of the Khalimsky space are ignored).
     * @param embedding
     * @return
     */
    template<typename T, typename result_type = typename T::value_type>
    auto
    khalimsky_2_contour2d(const xt::xexpression<T> &xkhalimsky, bool extra_border = false) {
        HG_TRACE();
        const auto &khalimsky = xkhalimsky.derived_cast();
        hg_assert(khalimsky.dimension() == 2, "Only 2d khalimsky grids are supported!");

        auto &shape = khalimsky.shape();

        long border = (extra_border) ? 0 : 1;

        std::array<long, 2> res_shape{(long) shape[0] / 2 + border, (long) shape[1] / 2 + border};
        embedding_grid_2d res_embedding(res_shape);

        auto g = get_4_adjacency_graph(res_embedding);
        array_1d <result_type> weights = xt::zeros<result_type>({num_edges(g)});

        point_2d_i one{{1, 1}};
        for (auto ei : edge_index_iterator(g)) {
            auto e = edge(ei, g);
            auto s = res_embedding.lin2grid(source(e, g));
            auto t = res_embedding.lin2grid(target(e, g));
            if (extra_border) {
                weights(ei) = khalimsky[s + t + one];
            } else {
                weights(ei) = khalimsky[s + t];
            }
        }

        return std::make_tuple(std::move(g), std::move(res_embedding), std::move(weights));
    };


    /**
     *
     */
    class contour_segment_2d {
        std::vector<index_t> m_contour_elements;

    public:

        contour_segment_2d(index_t contour_element) {
            m_contour_elements.push_back(contour_element);
        }

        template<typename T>
        contour_segment_2d(const T &contour_elements): m_contour_elements(std::begin(contour_elements),
                                                                          std::end(contour_elements)) {
        }

        const auto begin() const {
            return m_contour_elements.cbegin();
        }

        const auto end() const {
            return m_contour_elements.cend();
        }
    };

    class polyline_contour_2d {
        std::vector<contour_segment_2d> m_contour_segments;

    public:
        polyline_contour_2d() {

        }

        template<typename T>
        void add_segment(const T &contour_elements) {
            m_contour_segments.push_back(contour_elements);
        }

        const auto begin() const {
            return m_contour_segments.cbegin();
        }

        const auto end() const {
            return m_contour_segments.cend();
        }
    };

    class contour_2d {
        std::vector<polyline_contour_2d> m_polyline_contours;

    public:
        auto &new_polyline_contour_2d() {
            m_polyline_contours.emplace_back();
            return m_polyline_contours[m_polyline_contours.size() - 1];
        }

        auto size() {
            return m_polyline_contours.size();
        }

        const auto begin() const {
            return m_polyline_contours.begin();
        }

        const auto end() const {
            return m_polyline_contours.end();
        }
    };

    template<typename graph_t, typename T>
    auto
    fit_contour_2d(const graph_t &graph,
                   const embedding_grid_2d &embedding,
                   const xt::xexpression<T> &xedge_weights) {
        HG_TRACE();
        const auto &edge_weights = xedge_weights.derived_cast();
        hg_assert(edge_weights.dimension() == 1, "Edge weights must be scalar.");
        hg_assert(num_edges(graph) == edge_weights.size(),
                  "Edge weights size does not match the number of edge in the graph.");
        hg_assert(num_vertices(graph) == embedding.size(),
                  "Graph number of vertices does not match the size of the embedding.");

        contour_2d result;

        array_1d <index_t> positive_edge_index = xt::empty<index_t>({num_edges(graph)});
        for (index_t i = 0; i < positive_edge_index.size(); i++) {
            if (edge_weights[i] > 0)
                positive_edge_index[i] = i;
            else positive_edge_index[i] = invalid_index;
        }

        auto contours_khalimsky = contour2d_2_khalimsky(graph, embedding, positive_edge_index, true, invalid_index);

        array_2d<bool> processed = xt::zeros<bool>(contours_khalimsky.shape());

        auto height = contours_khalimsky.shape()[0];
        auto width = contours_khalimsky.shape()[1];

        auto is_intersection = [&contours_khalimsky, &height, &width](
                index_t y,
                index_t x) {
            if (x == 0 || y == 0 || x == width - 1 || y == height - 1)
                return true;
            int count = 0;
            if (contours_khalimsky(y, x - 1) != invalid_index)
                count++;
            if (contours_khalimsky(y, x + 1) != invalid_index)
                count++;
            if (contours_khalimsky(y - 1, x) != invalid_index)
                count++;
            if (contours_khalimsky(y + 1, x) != invalid_index)
                count++;
            return count > 2;
        };

        enum direction {
            NORTH, EAST, SOUTH, WEST
        };

        auto explore_contour_part = [&result, &contours_khalimsky, &processed, &is_intersection](
                index_t y,
                index_t x,
                direction dir) {
            auto &polyline = result.new_polyline_contour_2d();
            direction previous = dir;
            bool flag;

            do {
                processed(y, x) = true;
                index_t edge_index = contours_khalimsky(y, x);
                polyline.add_segment(edge_index);
                if (x % 2 == 0) // horizontal edge
                {
                    if (previous == NORTH) {
                        y++;
                    } else {
                        y--;
                    }
                } else { // vertical edge
                    if (previous == WEST) {
                        x++;
                    } else {
                        x--;
                    }
                }

                flag = is_intersection(y, x);
                if (!flag) {
                    processed(y, x) = true;
                    if (previous != NORTH &&
                        contours_khalimsky(y - 1, x) != invalid_index) {
                        previous = SOUTH;
                        y--;
                    } else if (previous != EAST &&
                               contours_khalimsky(y, x + 1) != invalid_index) {
                        previous = WEST;
                        x++;
                    } else if (previous != SOUTH &&
                               contours_khalimsky(y + 1, x) != invalid_index) {
                        previous = NORTH;
                        y++;
                    } else if (previous != WEST &&
                               contours_khalimsky(y, x - 1) != invalid_index) {
                        previous = EAST;
                        x--;
                    }
                }
            } while (!flag);

        };

        for (index_t y = 0; y < height; y += 2) {
            for (index_t x = 0; x < width; x += 2) {
                auto edge_index = contours_khalimsky(y, x);
                if (edge_index != invalid_index && // is there a non zero edge around this 0 face
                    !processed(y, x) && // if so did we already processed it ?
                    is_intersection(y, x)) {
                    processed(y, x) = true;
                    if (x != 0 && contours_khalimsky(y, x - 1) != invalid_index && !processed(y, x - 1)) {
                        explore_contour_part(y, x - 1, EAST);
                    }
                    if (x != width - 1 && contours_khalimsky(y, x + 1) != invalid_index && !processed(y, x + 1)) {
                        explore_contour_part(y, x + 1, WEST);
                    }
                    if (y != 0 && contours_khalimsky(y - 1, x) != invalid_index && !processed(y - 1, x)) {
                        explore_contour_part(y - 1, x, SOUTH);
                    }
                    if (y != height - 1 && contours_khalimsky(y + 1, x) != invalid_index && !processed(y + 1, x)) {
                        explore_contour_part(y + 1, x, NORTH);
                    }

                }
            }
        }

        return result;
    }
}
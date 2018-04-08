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
#include "array.hpp"


namespace hg {

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
    auto get_4_adjacency_implicit_graph(const embedding_grid_2d &embedding) {
        std::vector<point_2d_i> neighbours{{      -1, 0},
                                                 {0,  -1},
                                                 {0,  1},
                                                 {1,  0}}; // 4 adjacency

        return regular_grid_graph_2d(embedding, std::move(neighbours));
    }

    inline
    auto get_8_adjacency_implicit_graph(const embedding_grid_2d &embedding) {
        std::vector<point_2d_i> neighbours{{      -1, -1},
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

    enum class weight_functions {
        mean,
        min,
        max,
        L1,
        L2,
        L_infinity,
        L2_squared
    };

    template<typename graph_t, typename result_value_t=double>
    auto weight_graph(const graph_t &graph, const std::function<result_value_t(std::size_t, std::size_t)> &fun) {
        auto result = xt::xarray<result_value_t>::from_shape({num_edges(graph)});
        std::size_t i = 0;
        for (const auto e: edge_iterator(graph)) {
            result(i++) = fun(source(e, graph), target(e, graph));
        }
        return result;
    };


    template<typename graph_t, typename T, typename result_value_t=double>
    auto weight_graph(const graph_t &graph, const xt::xexpression<T> &xdata, weight_functions weight) {
        const auto &data = xdata.derived_cast();
        switch (weight) {
            case weight_functions::mean: {
                hg_assert(data.dimension() == 1, "Weight is only defined for scalar data.");
                std::function<result_value_t(std::size_t, std::size_t)> fun = [&graph, &data](std::size_t i,
                                                                                              std::size_t j)
                        -> result_value_t {
                    return static_cast<result_value_t>((data(i) + data(j)) / static_cast<result_value_t>(2.0));
                };
                return weight_graph(graph, fun);
            }
            case weight_functions::min: {
                hg_assert(data.dimension() == 1, "Weight is only defined for scalar data.");
                std::function<result_value_t(std::size_t, std::size_t)> fun = [&graph, &data](
                        std::size_t i, std::size_t j) -> result_value_t {
                    return std::min(data(i), data(j));
                };
                return weight_graph(graph, fun);
            }
            case weight_functions::max: {
                hg_assert(data.dimension() == 1, "Weight is only defined for scalar data.");
                std::function<result_value_t(std::size_t, std::size_t)> fun = [&graph, &data](std::size_t i,
                                                                                              std::size_t j) -> result_value_t {
                    return std::max(data(i), data(j));
                };
                return weight_graph(graph, fun);
            }
            case weight_functions::L1: {
                if (data.dimension() > 1) {
                    std::function<result_value_t(std::size_t, std::size_t)> fun = [&graph, &data](std::size_t i,
                                                                                                  std::size_t j) -> result_value_t {
                        return xt::sum(xt::abs(xt::dynamic_view(data, {i, xt::ellipsis()}) -
                                               xt::dynamic_view(data, {j, xt::ellipsis()})))();
                    };
                    return weight_graph(graph, fun);
                } else {
                    std::function<result_value_t(std::size_t, std::size_t)> fun = [&graph, &data](std::size_t i,
                                                                                                  std::size_t j) -> result_value_t {
                        return std::abs(data(i) - data(j));
                    };
                    return weight_graph(graph, fun);
                }
            }
            case weight_functions::L2: {
                if (data.dimension() > 1) {
                    std::function<result_value_t(std::size_t, std::size_t)> fun = [&graph, &data](std::size_t i,
                                                                                                  std::size_t j) -> result_value_t {
                        return std::sqrt(xt::sum(xt::pow(xt::dynamic_view(data, {i, xt::ellipsis()}) -
                                                         xt::dynamic_view(data, {j, xt::ellipsis()}), 2))());
                    };
                    return weight_graph(graph, fun);
                } else {
                    std::function<result_value_t(std::size_t, std::size_t)> fun = [&graph, &data](std::size_t i,
                                                                                                  std::size_t j) -> result_value_t {
                        auto v1 = data(i);
                        auto v2 = data(j);
                        return std::sqrt((v1 - v2) * (v1 - v2));
                    };
                    return weight_graph(graph, fun);
                }
            }
            case weight_functions::L_infinity: {
                if (data.dimension() > 1) {
                    std::function<result_value_t(std::size_t, std::size_t)> fun = [&graph, &data](std::size_t i,
                                                                                                  std::size_t j) -> result_value_t {
                        return xt::amax(xt::abs(xt::dynamic_view(data, {i, xt::ellipsis()}) -
                                                xt::dynamic_view(data, {j, xt::ellipsis()})))();
                    };
                    return weight_graph(graph, fun);
                } else {
                    std::function<result_value_t(std::size_t, std::size_t)> fun = [&graph, &data](std::size_t i,
                                                                                                  std::size_t j) -> result_value_t {
                        return std::abs(data(i) - data(j));
                    };
                    return weight_graph(graph, fun);
                }
            }
            case weight_functions::L2_squared: {
                if (data.dimension() > 1) {
                    std::function<result_value_t(std::size_t, std::size_t)> fun = [&graph, &data](std::size_t i,
                                                                                                  std::size_t j) -> result_value_t {
                        return xt::sum(xt::pow(xt::dynamic_view(data, {i, xt::ellipsis()}) -
                                               xt::dynamic_view(data, {j, xt::ellipsis()}), 2))();
                    };
                    return weight_graph(graph, fun);
                } else {
                    std::function<result_value_t(std::size_t, std::size_t)> fun = [&graph, &data](std::size_t i,
                                                                                                  std::size_t j) -> result_value_t {
                        auto v1 = data(i);
                        auto v2 = data(j);
                        return (v1 - v2) * (v1 - v2);
                    };
                    return weight_graph(graph, fun);
                }
            }

        }
        throw std::runtime_error("Unknown weight function.");

    };

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

        for (long y = 1; y < h - 1; ++y) {
            for (long x = 1; x < w - 1; ++x) {
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
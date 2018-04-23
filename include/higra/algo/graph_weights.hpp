//
// Created by user on 4/18/18.
//

#pragma once

#include "../graph.hpp"
#include "xtensor/xexpression.hpp"

namespace hg {
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
        auto result = array_1d<result_value_t>::from_shape({num_edges(graph)});
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
                        return xt::sum(xt::abs(xt::view(data, i) -
                                               xt::view(data, j)))();
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
                        return std::sqrt(xt::sum(xt::pow(xt::view(data, i) -
                                                         xt::view(data, j), 2))());
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
                        return xt::amax(xt::abs(xt::view(data, i) -
                                                xt::view(data, j)))();
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
                        return xt::sum(xt::pow(xt::view(data, i) -
                                               xt::view(data, j), 2))();
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
}
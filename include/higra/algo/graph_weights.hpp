//
// Created by user on 4/18/18.
//

#pragma once

#include "../graph.hpp"
#include "xtensor/xexpression.hpp"
#include "../structure/details/light_axis_view.hpp"

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
                std::function<result_value_t(std::size_t, std::size_t)> fun = [&data](std::size_t i,
                                                                                      std::size_t j)
                        -> result_value_t {
                    return static_cast<result_value_t>(
                            (static_cast<result_value_t>(data(i)) + static_cast<result_value_t>(data(j))) /
                            static_cast<result_value_t>(2.0));
                };
                return weight_graph(graph, fun);
            }
            case weight_functions::min: {
                hg_assert(data.dimension() == 1, "Weight is only defined for scalar data.");
                std::function<result_value_t(std::size_t, std::size_t)> fun = [&data](
                        std::size_t i, std::size_t j) -> result_value_t {
                    return std::min(data(i), data(j));
                };
                return weight_graph(graph, fun);
            }
            case weight_functions::max: {
                hg_assert(data.dimension() == 1, "Weight is only defined for scalar data.");
                std::function<result_value_t(std::size_t, std::size_t)> fun = [&data](std::size_t i,
                                                                                      std::size_t j) -> result_value_t {
                    return std::max(data(i), data(j));
                };
                return weight_graph(graph, fun);
            }
            case weight_functions::L1: {
                if (data.dimension() > 1) {
                    auto v1 = make_light_axis_view(data);
                    auto v2 = make_light_axis_view(data);
                    std::function<result_value_t(std::size_t, std::size_t)> fun = [&data, &v1, &v2](std::size_t i,
                                                                                                    std::size_t j) -> result_value_t {
                        v1.set_position(i);
                        v2.set_position(j);
                        result_value_t res = 0;
                        for (auto it1 = v1.begin(), it2 = v2.begin(); it1 != v1.end(); it1++, it2++) {
                            res += std::abs(static_cast<result_value_t>(*it1) - static_cast<result_value_t>(*it2));
                        }
                        return res;
                    };
                    return weight_graph(graph, fun);
                } else {
                    std::function<result_value_t(std::size_t, std::size_t)> fun = [&data](std::size_t i,
                                                                                          std::size_t j) -> result_value_t {
                        return std::abs(static_cast<result_value_t>(data(i)) - static_cast<result_value_t>(data(j)));
                    };
                    return weight_graph(graph, fun);
                }
            }
            case weight_functions::L2: {
                if (data.dimension() > 1) {
                    auto v1 = make_light_axis_view(data);
                    auto v2 = make_light_axis_view(data);
                    std::function<result_value_t(std::size_t, std::size_t)> fun = [&data, &v1, &v2](std::size_t i,
                                                                                                    std::size_t j) -> result_value_t {
                        v1.set_position(i);
                        v2.set_position(j);
                        result_value_t res = 0;
                        for (auto it1 = v1.begin(), it2 = v2.begin(); it1 != v1.end(); it1++, it2++) {
                            auto tmp = static_cast<result_value_t>(*it1) - static_cast<result_value_t>(*it2);
                            res += tmp * tmp;
                        }
                        return std::sqrt(res);
                    };
                    return weight_graph(graph, fun);
                } else {
                    std::function<result_value_t(std::size_t, std::size_t)> fun = [&data](std::size_t i,
                                                                                          std::size_t j) -> result_value_t {
                        auto v1 = data(i);
                        auto v2 = data(j);
                        auto tmp = static_cast<result_value_t>(v1) - static_cast<result_value_t>(v2);
                        return std::sqrt(tmp * tmp);
                    };
                    return weight_graph(graph, fun);
                }
            }
            case weight_functions::L_infinity: {
                if (data.dimension() > 1) {
                    auto v1 = make_light_axis_view(data);
                    auto v2 = make_light_axis_view(data);
                    std::function<result_value_t(std::size_t, std::size_t)> fun = [&data, &v1, &v2](std::size_t i,
                                                                                                    std::size_t j) -> result_value_t {
                        v1.set_position(i);
                        v2.set_position(j);
                        result_value_t res = -1;
                        for (auto it1 = v1.begin(), it2 = v2.begin(); it1 != v1.end(); it1++, it2++) {
                            res = std::max(res, std::abs(
                                    static_cast<result_value_t>(*it1) - static_cast<result_value_t>(*it2)));
                        }
                        return res;
                    };
                    return weight_graph(graph, fun);
                } else {
                    std::function<result_value_t(std::size_t, std::size_t)> fun = [&data](std::size_t i,
                                                                                          std::size_t j) -> result_value_t {
                        return std::abs(static_cast<result_value_t>(data(i)) - static_cast<result_value_t>(data(j)));
                    };
                    return weight_graph(graph, fun);
                }
            }
            case weight_functions::L2_squared: {
                if (data.dimension() > 1) {
                    auto v1 = make_light_axis_view(data);
                    auto v2 = make_light_axis_view(data);
                    std::function<result_value_t(std::size_t, std::size_t)> fun = [&data, &v1, &v2](std::size_t i,
                                                                                                    std::size_t j) -> result_value_t {
                        v1.set_position(i);
                        v2.set_position(j);
                        result_value_t res = 0;
                        for (auto it1 = v1.begin(), it2 = v2.begin(); it1 != v1.end(); it1++, it2++) {
                            auto tmp = static_cast<result_value_t>(*it1) - static_cast<result_value_t>(*it2);
                            res += tmp * tmp;
                        }
                        return res;
                    };
                    return weight_graph(graph, fun);
                } else {
                    std::function<result_value_t(std::size_t, std::size_t)> fun = [&graph, &data](std::size_t i,
                                                                                                  std::size_t j) -> result_value_t {
                        auto v1 = data(i);
                        auto v2 = data(j);
                        auto tmp = static_cast<result_value_t>(v1) - static_cast<result_value_t>(v2);
                        return tmp * tmp;
                    };
                    return weight_graph(graph, fun);
                }
            }

        }
        throw std::runtime_error("Unknown weight function.");

    };
}
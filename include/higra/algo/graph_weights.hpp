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
        L0,
        L1,
        L2,
        L_infinity,
        L2_squared
    };

    template<typename graph_t, typename result_value_t=double>
    auto weight_graph(const graph_t &graph, const std::function<result_value_t(
            typename graph_t::vertex_descriptor,
            typename graph_t::vertex_descriptor)> &fun) {
        auto result = array_1d<result_value_t>::from_shape({num_edges(graph)});
        index_t i = 0;
        for (const auto e: edge_iterator(graph)) {
            result(i++) = fun(source(e, graph), target(e, graph));
        }
        return result;
    };


    template<typename graph_t, typename T, typename result_value_t=double>
    auto weight_graph(const graph_t &graph, const xt::xexpression<T> &xvertex_weights, weight_functions weight) {
        HG_TRACE();
        using vertex_t = typename graph_t::vertex_descriptor;
        const auto &vertex_weights = xvertex_weights.derived_cast();
        hg_assert(vertex_weights.dimension() > 0, "Vertex weights cannot have 0 dimension.");
        hg_assert(num_vertices(graph) == vertex_weights.shape()[0],
                  "Vertex weights dimension does not match graph number of vertices.");

        switch (weight) {
            case weight_functions::mean: {
                hg_assert(vertex_weights.dimension() == 1, "Weight is only defined for scalar data.");
                std::function<result_value_t(vertex_t, vertex_t)> fun = [&vertex_weights](vertex_t i,
                                                                                          vertex_t j)
                        -> result_value_t {
                    return static_cast<result_value_t>(
                            (static_cast<result_value_t>(vertex_weights(i)) +
                             static_cast<result_value_t>(vertex_weights(j))) /
                            static_cast<result_value_t>(2.0));
                };
                return weight_graph(graph, fun);
            }
            case weight_functions::min: {
                hg_assert(vertex_weights.dimension() == 1, "Weight is only defined for scalar data.");
                std::function<result_value_t(vertex_t, vertex_t)> fun = [&vertex_weights](
                        vertex_t i, vertex_t j) -> result_value_t {
                    return std::min(vertex_weights(i), vertex_weights(j));
                };
                return weight_graph(graph, fun);
            }
            case weight_functions::max: {
                hg_assert(vertex_weights.dimension() == 1, "Weight is only defined for scalar data.");
                std::function<result_value_t(vertex_t, vertex_t)> fun = [&vertex_weights](vertex_t i,
                                                                                          vertex_t j) -> result_value_t {
                    return std::max(vertex_weights(i), vertex_weights(j));
                };
                return weight_graph(graph, fun);
            }
            case weight_functions::L0: {
                if (vertex_weights.dimension() > 1) {
                    auto v1 = make_light_axis_view(vertex_weights);
                    auto v2 = make_light_axis_view(vertex_weights);
                    std::function<result_value_t(vertex_t, vertex_t)> fun = [&v1, &v2](
                            vertex_t i,
                            vertex_t j) -> result_value_t {
                        v1.set_position(i);
                        v2.set_position(j);
                        for (auto it1 = v1.begin(), it2 = v2.begin(); it1 != v1.end(); it1++, it2++) {
                            if (*it1 != *it2)
                                return 1;
                        }
                        return 0;
                    };
                    return weight_graph(graph, fun);
                } else {
                    std::function<result_value_t(vertex_t, vertex_t)> fun = [&vertex_weights](vertex_t i,
                                                                                              vertex_t j) -> result_value_t {
                        return (vertex_weights(i) == vertex_weights(j)) ? 0 : 1;
                    };
                    return weight_graph(graph, fun);
                }
            }
            case weight_functions::L1: {
                if (vertex_weights.dimension() > 1) {
                    auto v1 = make_light_axis_view(vertex_weights);
                    auto v2 = make_light_axis_view(vertex_weights);
                    std::function<result_value_t(vertex_t, vertex_t)> fun = [&v1, &v2](
                            vertex_t i,
                            vertex_t j) -> result_value_t {
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
                    std::function<result_value_t(vertex_t, vertex_t)> fun = [&vertex_weights](vertex_t i,
                                                                                              vertex_t j) -> result_value_t {
                        return std::abs(static_cast<result_value_t>(vertex_weights(i)) -
                                        static_cast<result_value_t>(vertex_weights(j)));
                    };
                    return weight_graph(graph, fun);
                }
            }
            case weight_functions::L2: {
                if (vertex_weights.dimension() > 1) {
                    auto v1 = make_light_axis_view(vertex_weights);
                    auto v2 = make_light_axis_view(vertex_weights);
                    std::function<result_value_t(vertex_t, vertex_t)> fun = [&v1, &v2](
                            vertex_t i,
                            vertex_t j) -> result_value_t {
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
                    std::function<result_value_t(vertex_t, vertex_t)> fun = [&vertex_weights](vertex_t i,
                                                                                              vertex_t j) -> result_value_t {
                        auto v1 = vertex_weights(i);
                        auto v2 = vertex_weights(j);
                        auto tmp = static_cast<result_value_t>(v1) - static_cast<result_value_t>(v2);
                        return std::sqrt(tmp * tmp);
                    };
                    return weight_graph(graph, fun);
                }
            }
            case weight_functions::L_infinity: {
                if (vertex_weights.dimension() > 1) {
                    auto v1 = make_light_axis_view(vertex_weights);
                    auto v2 = make_light_axis_view(vertex_weights);
                    std::function<result_value_t(vertex_t, vertex_t)> fun = [&v1, &v2](
                            vertex_t i,
                            vertex_t j) -> result_value_t {
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
                    std::function<result_value_t(vertex_t, vertex_t)> fun = [&vertex_weights](vertex_t i,
                                                                                              vertex_t j) -> result_value_t {
                        return std::abs(static_cast<result_value_t>(vertex_weights(i)) -
                                        static_cast<result_value_t>(vertex_weights(j)));
                    };
                    return weight_graph(graph, fun);
                }
            }
            case weight_functions::L2_squared: {
                if (vertex_weights.dimension() > 1) {
                    auto v1 = make_light_axis_view(vertex_weights);
                    auto v2 = make_light_axis_view(vertex_weights);
                    std::function<result_value_t(vertex_t, vertex_t)> fun = [&v1, &v2](
                            vertex_t i,
                            vertex_t j) -> result_value_t {
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
                    std::function<result_value_t(vertex_t, vertex_t)> fun = [&vertex_weights](
                            vertex_t i,
                            vertex_t j) -> result_value_t {
                        auto v1 = vertex_weights(i);
                        auto v2 = vertex_weights(j);
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
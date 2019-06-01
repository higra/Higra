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
#include "xtensor/xexpression.hpp"
#include "../structure/details/light_axis_view.hpp"

namespace hg {

    /**
     * Predefined edge-weighting functions (see weight_graph function)
     */
    enum class weight_functions {
        mean,
        min,
        max,
        L0,
        L1,
        L2,
        L_infinity,
        L2_squared,
        source,
        target
    };

    /**
     * Compute edge-weights of a graph based on a weighting function.
     *
     * A weighting function is a function that associates a weights to a pair of vertices.
     *
     * @tparam graph_t
     * @tparam result_value_t
     * @param graph
     * @param fun
     * @return an array of weights
     */
    template<typename result_value_t=double, typename graph_t>
    auto weight_graph(const graph_t &graph, const std::function<result_value_t(
            typename graph_t::vertex_descriptor,
            typename graph_t::vertex_descriptor)> &fun) {
        auto result = array_1d<result_value_t>::from_shape({num_edges(graph)});

        for (const auto e: edge_iterator(graph)) {
            result(e) = fun(source(e, graph), target(e, graph));
        }
        return result;
    };

    /**
     * Compute edge-weights of a graph based from the vertex-weights and a predefined weighting function (see weight_functions enum).
     *
     * Each edge is weighted with a combination of its extremities weights.
     *
     * @tparam result_value_t The value type of the result
     * @tparam promoted_type The value type used for internal computation
     * @tparam graph_t
     * @tparam T
     * @param graph
     * @param xvertex_weights
     * @param weight
     * @return
     */
    template<typename result_value_t = double,
            typename promoted_type = double,
            typename graph_t,
            typename T>
    auto weight_graph(const graph_t &graph, const xt::xexpression<T> &xvertex_weights, weight_functions weight) {
        HG_TRACE();
        using vertex_t = typename graph_t::vertex_descriptor;
        const auto &vertex_weights = xvertex_weights.derived_cast();
        hg_assert_vertex_weights(graph, vertex_weights);

        switch (weight) {
            case weight_functions::mean: {
                hg_assert_1d_array(vertex_weights);
                std::function<result_value_t(vertex_t, vertex_t)> fun = [&vertex_weights](vertex_t i,
                                                                                          vertex_t j)
                        -> result_value_t {
                    return static_cast<result_value_t>(
                            (static_cast<promoted_type>(vertex_weights(i)) +
                             static_cast<promoted_type>(vertex_weights(j))) /
                            static_cast<promoted_type>(2.0));
                };
                return weight_graph(graph, fun);
            }
            case weight_functions::min: {
                hg_assert_1d_array(vertex_weights);
                std::function<result_value_t(vertex_t, vertex_t)> fun = [&vertex_weights](
                        vertex_t i, vertex_t j) -> result_value_t {
                    return static_cast<result_value_t>(std::min(vertex_weights(i), vertex_weights(j)));
                };
                return weight_graph(graph, fun);
            }
            case weight_functions::max: {
                hg_assert_1d_array(vertex_weights);
                std::function<result_value_t(vertex_t, vertex_t)> fun = [&vertex_weights](vertex_t i,
                                                                                          vertex_t j) -> result_value_t {
                    return static_cast<result_value_t>(std::max(vertex_weights(i), vertex_weights(j)));
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
                        promoted_type res = 0;
                        for (auto it1 = v1.begin(), it2 = v2.begin(); it1 != v1.end(); it1++, it2++) {
                            res += std::abs(static_cast<promoted_type>(*it1) - static_cast<promoted_type>(*it2));
                        }
                        return static_cast<result_value_t>(res);
                    };
                    return weight_graph(graph, fun);
                } else {
                    std::function<result_value_t(vertex_t, vertex_t)> fun = [&vertex_weights](vertex_t i,
                                                                                              vertex_t j) -> result_value_t {
                        return static_cast<result_value_t>(std::abs(static_cast<promoted_type>(vertex_weights(i)) -
                                        static_cast<promoted_type>(vertex_weights(j))));
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
                        promoted_type res = 0;
                        for (auto it1 = v1.begin(), it2 = v2.begin(); it1 != v1.end(); it1++, it2++) {
                            auto tmp = static_cast<promoted_type>(*it1) - static_cast<promoted_type>(*it2);
                            res += tmp * tmp;
                        }
                        return static_cast<result_value_t>(std::sqrt(res));
                    };
                    return weight_graph(graph, fun);
                } else {
                    std::function<result_value_t(vertex_t, vertex_t)> fun = [&vertex_weights](vertex_t i,
                                                                                              vertex_t j) -> result_value_t {
                        auto v1 = vertex_weights(i);
                        auto v2 = vertex_weights(j);
                        auto tmp = static_cast<promoted_type>(v1) - static_cast<promoted_type>(v2);
                        return static_cast<result_value_t>(std::sqrt(tmp * tmp));
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
                        promoted_type res = -1;
                        for (auto it1 = v1.begin(), it2 = v2.begin(); it1 != v1.end(); it1++, it2++) {
                            res = std::max(res, std::abs(
                                    static_cast<promoted_type>(*it1) - static_cast<promoted_type>(*it2)));
                        }
                        return static_cast<result_value_t>(res);
                    };
                    return weight_graph(graph, fun);
                } else {
                    std::function<result_value_t(vertex_t, vertex_t)> fun = [&vertex_weights](vertex_t i,
                                                                                              vertex_t j) -> result_value_t {
                        return static_cast<result_value_t>(std::abs(static_cast<promoted_type>(vertex_weights(i)) -
                                        static_cast<promoted_type>(vertex_weights(j))));
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
                        promoted_type res = 0;
                        for (auto it1 = v1.begin(), it2 = v2.begin(); it1 != v1.end(); it1++, it2++) {
                            auto tmp = static_cast<promoted_type>(*it1) - static_cast<promoted_type>(*it2);
                            res += tmp * tmp;
                        }
                        return static_cast<result_value_t>(res);
                    };
                    return weight_graph(graph, fun);
                } else {
                    std::function<result_value_t(vertex_t, vertex_t)> fun = [&vertex_weights](
                            vertex_t i,
                            vertex_t j) -> result_value_t {
                        auto v1 = vertex_weights(i);
                        auto v2 = vertex_weights(j);
                        auto tmp = static_cast<promoted_type>(v1) - static_cast<promoted_type>(v2);
                        return static_cast<result_value_t>(tmp * tmp);
                    };
                    return weight_graph(graph, fun);
                }
            }
            case weight_functions::source: {
                hg_assert_1d_array(vertex_weights);
                std::function<result_value_t(vertex_t, vertex_t)> fun = [&vertex_weights](vertex_t i,
                                                                                          vertex_t j) -> result_value_t {
                    return static_cast<result_value_t>(vertex_weights(i));
                };
                return weight_graph(graph, fun);
            }
            case weight_functions::target: {
                hg_assert_1d_array(vertex_weights);
                std::function<result_value_t(vertex_t, vertex_t)> fun = [&vertex_weights](vertex_t i,
                                                                                          vertex_t j) -> result_value_t {
                    return static_cast<result_value_t>(vertex_weights(j));
                };
                return weight_graph(graph, fun);
            }
        }
        throw std::runtime_error("Unknown weight function.");
    };
}
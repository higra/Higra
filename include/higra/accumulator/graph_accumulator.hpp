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
#include "accumulator.hpp"
#include "../structure/details/light_axis_view.hpp"

namespace hg {

    namespace graph_accumulator_detail {


        template<bool vectorial,
                typename graph_t,
                typename T,
                typename accumulator_t,
                typename output_t = typename T::value_type>
        auto accumulate_graph_edges_impl(const graph_t &graph,
                                         const xt::xexpression<T> &xinput,
                                         const accumulator_t accumulator) {
            HG_TRACE();
            auto &input = xinput.derived_cast();
            hg_assert_edge_weights(graph, input);

            auto data_shape = std::vector<size_t>(input.shape().begin() + 1, input.shape().end());
            auto output_shape = accumulator_t::get_output_shape(data_shape);
            output_shape.insert(output_shape.begin(), num_vertices(graph));

            array_nd<output_t> output = array_nd<output_t>::from_shape(output_shape);

            auto input_view = make_light_axis_view<vectorial>(input);
            auto output_view = make_light_axis_view<vectorial>(output);
            auto acc = accumulator.template make_accumulator<vectorial>(output_view);

            for (auto i: vertex_iterator(graph)) {
                output_view.set_position(i);
                acc.set_storage(output_view);
                acc.initialize();
                for (auto e: out_edge_iterator(i, graph)) {
                    input_view.set_position(e);
                    acc.accumulate(input_view.begin());
                }
            }

            return output;
        };

        template<bool vectorial,
                typename graph_t,
                typename T,
                typename accumulator_t,
                typename output_t = typename T::value_type>
        auto accumulate_graph_vertices_impl(const graph_t &graph,
                                            const xt::xexpression<T> &xinput,
                                            const accumulator_t accumulator) {
            HG_TRACE();
            auto &input = xinput.derived_cast();
            hg_assert_vertex_weights(graph, input);

            auto data_shape = std::vector<size_t>(input.shape().begin() + 1, input.shape().end());
            auto output_shape = accumulator_t::get_output_shape(data_shape);
            output_shape.insert(output_shape.begin(), num_vertices(graph));

            array_nd<output_t> output = array_nd<output_t>::from_shape(output_shape);

            auto input_view = make_light_axis_view<vectorial>(input);
            auto output_view = make_light_axis_view<vectorial>(output);
            auto acc = accumulator.template make_accumulator<vectorial>(output_view);

            for (auto i: vertex_iterator(graph)) {
                output_view.set_position(i);
                acc.set_storage(output_view);
                acc.initialize();
                for (auto v: adjacent_vertex_iterator(i, graph)) {
                    input_view.set_position(v);
                    acc.accumulate(input_view.begin());
                }
            }

            return output;
        };


    }

    template<typename graph_t, typename T, typename accumulator_t, typename output_t = typename T::value_type>
    auto accumulate_graph_edges(const graph_t &graph,
                                const xt::xexpression<T> &xedge_weights,
                                const accumulator_t &accumulator) {
        auto &edge_weights = xedge_weights.derived_cast();
        if (edge_weights.dimension() == 1) {
            return graph_accumulator_detail::accumulate_graph_edges_impl<false>(graph, xedge_weights, accumulator);
        } else {
            return graph_accumulator_detail::accumulate_graph_edges_impl<true>(graph, xedge_weights, accumulator);
        }
    };

    template<typename graph_t, typename T, typename accumulator_t, typename output_t = typename T::value_type>
    auto accumulate_graph_vertices(const graph_t &graph,
                                   const xt::xexpression<T> &xvertex_weights,
                                   const accumulator_t &accumulator) {
        auto &vertex_weights = xvertex_weights.derived_cast();
        if (vertex_weights.dimension() == 1) {
            return graph_accumulator_detail::accumulate_graph_vertices_impl<false>(graph, xvertex_weights, accumulator);
        } else {
            return graph_accumulator_detail::accumulate_graph_vertices_impl<true>(graph, xvertex_weights, accumulator);
        }
    };


}
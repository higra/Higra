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
                typename V,
                typename accumulator_t,
                typename output_t = accumulator_output_value_t<accumulator_t, V>>
        auto accumulate_graph_edges_impl(const graph_t &graph,
                                         array_view<const V> input,
                                         const accumulator_t accumulator) {
            HG_TRACE();
            hg_assert_edge_weights(graph, input);

            auto data_shape = std::vector<size_t>(input.shape().begin() + 1, input.shape().end());
            const index_t input_row_size = details::light_axis_view_row_size(input);
            auto output_shape = accumulator_t::get_output_shape(data_shape);
            output_shape.insert(output_shape.begin(), num_vertices(graph));

            array_nd<output_t> output = array_nd<output_t>::from_shape(array_shape::from(output_shape));

            auto input_view = make_light_axis_view<vectorial>(input);
            auto output_view = make_light_axis_view<vectorial>(output);
            auto acc = accumulator_detail::make_accumulator<vectorial, V>(accumulator, output_view, input_row_size);

            for (auto i: vertex_iterator(graph)) {
                output_view.set_position(i);
                acc.set_storage(output_view);
                acc.initialize();
                for (auto e: out_edge_iterator(i, graph)) {
                    input_view.set_position(e);
                    acc.accumulate(input_view.begin());
                }
                acc.finalize();
            }

            return output;
        };

        template<bool vectorial,
                typename graph_t,
                typename V,
                typename accumulator_t,
                typename output_t = accumulator_output_value_t<accumulator_t, V>>
        auto accumulate_graph_vertices_impl(const graph_t &graph,
                                            array_view<const V> input,
                                            const accumulator_t accumulator) {
            HG_TRACE();
            hg_assert_vertex_weights(graph, input);

            auto data_shape = std::vector<size_t>(input.shape().begin() + 1, input.shape().end());
            const index_t input_row_size = details::light_axis_view_row_size(input);
            auto output_shape = accumulator_t::get_output_shape(data_shape);
            output_shape.insert(output_shape.begin(), num_vertices(graph));

            array_nd<output_t> output = array_nd<output_t>::from_shape(array_shape::from(output_shape));

            auto input_view = make_light_axis_view<vectorial>(input);
            auto output_view = make_light_axis_view<vectorial>(output);
            auto acc = accumulator_detail::make_accumulator<vectorial, V>(accumulator, output_view, input_row_size);

            for (auto i: vertex_iterator(graph)) {
                output_view.set_position(i);
                acc.set_storage(output_view);
                acc.initialize();
                for (auto v: adjacent_vertex_iterator(i, graph)) {
                    input_view.set_position(v);
                    acc.accumulate(input_view.begin());
                }
                acc.finalize();
            }

            return output;
        };


    }

    template<typename graph_t, typename T, typename accumulator_t, typename output_t = T>
    auto accumulate_graph_edges(const graph_t &graph,
                                array_view<const T> edge_weights,
                                const accumulator_t &accumulator) {
        if (edge_weights.dimension() == 1) {
            return graph_accumulator_detail::accumulate_graph_edges_impl<false>(graph, edge_weights, accumulator);
        } else {
            return graph_accumulator_detail::accumulate_graph_edges_impl<true>(graph, edge_weights, accumulator);
        }
    };

    template<typename graph_t, typename T, typename accumulator_t, typename output_t = T>
    auto accumulate_graph_vertices(const graph_t &graph,
                                   array_view<const T> vertex_weights,
                                   const accumulator_t &accumulator) {
        if (vertex_weights.dimension() == 1) {
            return graph_accumulator_detail::accumulate_graph_vertices_impl<false>(graph, vertex_weights, accumulator);
        } else {
            return graph_accumulator_detail::accumulate_graph_vertices_impl<true>(graph, vertex_weights, accumulator);
        }
    };


}
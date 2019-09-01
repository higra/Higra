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

    namespace tree_contour_accumulator_detail {

        template<bool vectorial,
                typename graph_t,
                typename tree_t,
                typename T,
                typename T2,
                typename accumulator_t,
                typename output_t = typename T::value_type>
        auto accumulate_on_contours_impl(const graph_t &graph,
                                         const tree_t &tree,
                                         const xt::xexpression<T> &xinput,
                                         const xt::xexpression<T2> &xdetph,
                                         const accumulator_t accumulator) {
            HG_TRACE();
            auto &input = xinput.derived_cast();
            hg_assert_node_weights(tree, input);
            auto &depth = xdetph.derived_cast();
            hg_assert_node_weights(tree, depth);
            hg_assert_1d_array(depth);
            hg_assert_integral_value_type(depth);

            auto data_shape = std::vector<size_t>(input.shape().begin() + 1, input.shape().end());
            auto output_shape = accumulator_t::get_output_shape(data_shape);
            output_shape.insert(output_shape.begin(), num_edges(graph));

            array_nd<output_t> output = array_nd<output_t>::from_shape(output_shape);

            auto input_view = make_light_axis_view<vectorial>(input);
            auto output_view = make_light_axis_view<vectorial>(output);
            auto acc = accumulator.template make_accumulator<vectorial>(output_view);

            for (auto e: edge_iterator(graph)) {
                auto n1 = source(e, graph);
                auto n2 = target(e, graph);
                auto i = index(e, graph);

                output_view.set_position(i);
                acc.set_storage(output_view);
                acc.initialize();

                while (n1 != n2) {
                    auto dn1 = depth(n1);
                    auto dn2 = depth(n2);
                    auto new_n1 = n1;
                    auto new_n2 = n2;
                    if (dn1 >= dn2) {
                        input_view.set_position(n1);
                        acc.accumulate(input_view.begin());
                        new_n1 = parent(n1, tree);
                    }
                    if (dn2 >= dn1) {
                        input_view.set_position(n2);
                        acc.accumulate(input_view.begin());
                        new_n2 = parent(n2, tree);
                    }
                    n1 = new_n1;
                    n2 = new_n2;
                }
                acc.finalize();
            }

            return output;
        };

    }

    template<typename graph_t, typename tree_t, typename T, typename T1, typename accumulator_t, typename output_t = typename T::value_type>
    auto accumulate_on_contours(const graph_t &graph,
                                const tree_t &tree,
                                const xt::xexpression<T> &xinput,
                                const xt::xexpression<T1> &xdepth,
                                const accumulator_t &accumulator) {
        auto &input = xinput.derived_cast();
        if (input.dimension() == 1) {
            return tree_contour_accumulator_detail::accumulate_on_contours_impl<false>(graph,
                                                                                       tree,
                                                                                       xinput,
                                                                                       xdepth,
                                                                                       accumulator);
        } else {
            return tree_contour_accumulator_detail::accumulate_on_contours_impl<true>(graph,
                                                                                      tree,
                                                                                      xinput,
                                                                                      xdepth,
                                                                                      accumulator);
        }
    };

}
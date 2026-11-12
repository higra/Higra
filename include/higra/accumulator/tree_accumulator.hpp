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
#include "../structure/array/xtensor_bridge.hpp"

namespace hg {

    namespace tree_accumulator_detail {


        template<bool vectorial,
                typename tree_t,
                typename V,
                typename accumulator_t,
                typename output_t = V>
        auto accumulate_parallel_impl(const tree_t &tree,
                                      array_view<const V> input,
                                      const accumulator_t accumulator) {
            HG_TRACE();
            hg_assert_node_weights(tree, input);

            auto data_shape = std::vector<size_t>(input.shape().begin() + 1, input.shape().end());
            auto output_shape = accumulator_t::get_output_shape(data_shape);
            output_shape.insert(output_shape.begin(), num_vertices(tree));

            array_nd <output_t> output = array_nd<output_t>::from_shape(output_shape);

            auto input_view = make_light_axis_view<vectorial>(input);
            auto output_view = make_light_axis_view<vectorial>(output);

            auto acc = accumulator.template make_accumulator<vectorial>(output_view);

            for (auto i: leaves_iterator(tree)) {
                output_view.set_position(i);
                acc.set_storage(output_view);
                acc.initialize();
                acc.finalize();
            }

            if (tree.children_computed()) {

                for (auto i : leaves_to_root_iterator(tree, leaves_it::exclude)) {
                    output_view.set_position(i);
                    acc.set_storage(output_view);
                    acc.initialize();
                    for (auto c : children_iterator(i, tree)) {
                        input_view.set_position(c);
                        acc.accumulate(input_view.begin());
                    }
                    acc.finalize();
                }

            } else {
                index_t numl = num_leaves(tree);
                std::vector<decltype(accumulator.template make_accumulator<vectorial>(output_view))> accs;
                accs.reserve(num_vertices(tree) - numl);

                for (auto i : leaves_to_root_iterator(tree, leaves_it::exclude)) {
                    output_view.set_position(i);
                    accs.push_back(accumulator.template make_accumulator<vectorial>(output_view));
                    accs.back().initialize();
                }

                for (auto i : leaves_to_root_iterator(tree, leaves_it::include, root_it::exclude)) {
                    if (i >= numl) {
                        accs[i - numl].finalize();
                    }

                    auto p = parent(i, tree);
                    input_view.set_position(i);
                    accs[p - numl].accumulate(input_view.begin());
                }
                accs.back().finalize();
            }

            return output;
        };

        template<bool vectorial,
                typename tree_t,
                typename V,
                typename accumulator_t,
                typename output_t = V>
        auto accumulate_sequential_impl(const tree_t &tree,
                                        array_view<const V> vertex_data,
                                        const accumulator_t &accumulator) {
            HG_TRACE();
            hg_assert_leaf_weights(tree, vertex_data);

            auto data_shape = std::vector<size_t>(vertex_data.shape().begin() + 1, vertex_data.shape().end());
            auto output_shape = accumulator_t::get_output_shape(data_shape);
            output_shape.insert(output_shape.begin(), num_vertices(tree));

            array_nd <output_t> output = array_nd<output_t>::from_shape(output_shape);

            auto vertex_data_view = make_light_axis_view<vectorial>(vertex_data);
            auto input_view = make_light_axis_view<vectorial>(output);
            auto output_view = make_light_axis_view<vectorial>(output);

            for (auto i: leaves_iterator(tree)) {
                output_view.set_position(i);
                vertex_data_view.set_position(i);
                output_view = vertex_data_view;
            }

            if (tree.children_computed()) {
                auto acc = accumulator.template make_accumulator<vectorial>(output_view);

                for (auto i : leaves_to_root_iterator(tree, leaves_it::exclude)) {
                    output_view.set_position(i);
                    acc.set_storage(output_view);
                    acc.initialize();
                    for (auto c : children_iterator(i, tree)) {
                        input_view.set_position(c);
                        acc.accumulate(input_view.begin());
                    }
                    acc.finalize();
                }
            } else {
                index_t numl = num_leaves(tree);
                std::vector<decltype(accumulator.template make_accumulator<vectorial>(output_view))> accs;
                accs.reserve(num_vertices(tree) - numl);

                for (auto i : leaves_to_root_iterator(tree, leaves_it::exclude)) {
                    output_view.set_position(i);
                    accs.push_back(accumulator.template make_accumulator<vectorial>(output_view));
                    accs.back().initialize();
                }

                for (auto i : leaves_to_root_iterator(tree, leaves_it::include, root_it::exclude)) {
                    if (i >= numl) {
                        accs[i - numl].finalize();
                    }

                    auto p = parent(i, tree);
                    input_view.set_position(i);
                    accs[p - numl].accumulate(input_view.begin());
                }
                accs.back().finalize();
            }

            return output;
        };

        template<bool vectorial,
                typename tree_t,
                typename V1,
                typename V2,
                typename accumulator_t,
                typename combination_fun_t,
                typename output_t = V1>
        auto accumulate_and_combine_sequential_impl(const tree_t &tree,
                                                    array_view<const V1> input,
                                                    array_view<const V2> vertex_data,
                                                    accumulator_t &accumulator,
                                                    combination_fun_t combine) {
            HG_TRACE();
            hg_assert_node_weights(tree, input);

            hg_assert_leaf_weights(tree, vertex_data);

            auto data_shape = std::vector<size_t>(input.shape().begin() + 1, input.shape().end());
            auto output_shape = accumulator_t::get_output_shape(data_shape);
            hg_assert((index_t) output_shape.size() == input.dimension() - 1,
                      "Input dimension does not match accumulator output dimension.");
            hg_assert((index_t) output_shape.size() == vertex_data.dimension() - 1,
                      "Vertex data dimension does not match accumulator output dimension.");
            hg_assert(std::equal(output_shape.begin(), output_shape.end(), input.shape().begin() + 1),
                      "Input shape does not match accumulator output shape.");
            hg_assert(std::equal(output_shape.begin(), output_shape.end(), vertex_data.shape().begin() + 1),
                      "Vertex data shape does not match accumulator output shape.");
            output_shape.insert(output_shape.begin(), num_vertices(tree));

            array_nd <output_t> output = array_nd<output_t>::from_shape(output_shape);

            auto input_view = make_light_axis_view<vectorial>(input);
            auto inout_view = make_light_axis_view<vectorial>(output);
            auto output_view = make_light_axis_view<vectorial>(output);

            auto vertex_data_view = make_light_axis_view<vectorial>(vertex_data);

            for (auto i: leaves_iterator(tree)) {
                output_view.set_position(i);
                vertex_data_view.set_position(i);
                output_view = vertex_data_view;
            }

            if (tree.children_computed()) {
                auto acc = accumulator.template make_accumulator<vectorial>(output_view);

                for (auto i : leaves_to_root_iterator(tree, leaves_it::exclude)) {
                    output_view.set_position(i);
                    acc.set_storage(output_view);
                    acc.initialize();
                    for (auto c : children_iterator(i, tree)) {
                        inout_view.set_position(c);
                        acc.accumulate(inout_view.begin());
                    }
                    acc.finalize();
                    input_view.set_position(i);
                    output_view.combine(input_view, combine);
                }
            } else {
                index_t numl = num_leaves(tree);
                std::vector<decltype(accumulator.template make_accumulator<vectorial>(output_view))> accs;
                accs.reserve(num_vertices(tree) - numl);

                for (auto i : leaves_to_root_iterator(tree, leaves_it::exclude)) {
                    output_view.set_position(i);
                    accs.push_back(accumulator.template make_accumulator<vectorial>(output_view));
                    accs.back().initialize();
                }

                for (auto i : leaves_to_root_iterator(tree, leaves_it::include, root_it::exclude)) {
                    if (i >= numl) {
                        accs[i - numl].finalize();
                        input_view.set_position(i);
                        output_view.set_position(i);
                        output_view.combine(input_view, combine);
                    }

                    auto p = parent(i, tree);
                    inout_view.set_position(i);
                    accs[p - numl].accumulate(inout_view.begin());
                }

                accs.back().finalize();
                input_view.set_position(root(tree));
                output_view.set_position(root(tree));
                output_view.combine(input_view, combine);
            }

            return output;
        };

        template<bool vectorial,
                typename tree_t,
                typename V1,
                typename output_t = V1>
        auto propagate_parallel_impl(const tree_t &tree,
                                     array_view<const V1> input) {
            HG_TRACE();
            hg_assert_node_weights(tree, input);

            array_nd <output_t> output = array_nd<output_t>::from_shape(
                    std::vector<size_t>(input.shape().begin(), input.shape().end()));

            auto input_view = make_light_axis_view<vectorial>(input);
            auto output_view = make_light_axis_view<vectorial>(output);

            auto aparents = parents(tree).data();

            for (auto i: root_to_leaves_iterator(tree)) {
                input_view.set_position(aparents[i]);
                output_view.set_position(i);
                output_view = input_view;
            }
            return output;
        };

        template<bool vectorial,
                typename tree_t,
                typename V1,
                typename T2,
                typename output_t = V1>
        auto propagate_parallel_impl(const tree_t &tree,
                                     array_view<const V1> input,
                                     const xt::xexpression<T2> &xcondition) {
            HG_TRACE();
            auto &condition = xcondition.derived_cast();
            hg_assert_node_weights(tree, input);
            hg_assert_node_weights(tree, condition);

            array_nd <output_t> output = array_nd<output_t>::from_shape(
                    std::vector<size_t>(input.shape().begin(), input.shape().end()));

            auto input_view = make_light_axis_view<vectorial>(input);
            auto output_view = make_light_axis_view<vectorial>(output);

            auto aparents = parents(tree).data();

            for (auto i: root_to_leaves_iterator(tree)) {
                if (condition(i)) {
                    input_view.set_position(aparents[i]);
                } else {
                    input_view.set_position(i);
                }
                output_view.set_position(i);
                output_view = input_view;
            }
            return output;
        };

        template<bool vectorial,
                typename tree_t,
                typename V1,
                typename T2,
                typename output_t = V1>
        auto propagate_sequential_impl(const tree_t &tree,
                                       array_view<const V1> input,
                                       const xt::xexpression<T2> &xcondition) {
            HG_TRACE();
            auto &condition = xcondition.derived_cast();
            hg_assert_node_weights(tree, input);
            hg_assert_node_weights(tree, condition);
            hg_assert_1d_array(condition);

            array_nd <output_t> output = array_nd<output_t>::from_shape(
                    std::vector<size_t>(input.shape().begin(), input.shape().end()));

            auto input_view = make_light_axis_view<vectorial>(input);
            auto output_view = make_light_axis_view<vectorial>(output);
            auto inout_view = make_light_axis_view<vectorial>(output);

            auto aparents = parents(tree).data();

            // root cannot be deleted
            output_view.set_position(root(tree));
            input_view.set_position(root(tree));
            output_view = input_view;

            for (auto i: root_to_leaves_iterator(tree, leaves_it::include, root_it::exclude)) {
                output_view.set_position(i);
                if (condition(i)) {
                    inout_view.set_position(aparents[i]);
                    output_view = inout_view;
                } else {
                    input_view.set_position(i);
                    output_view = input_view;
                }

            }
            return output;
        };

        template<bool vectorial,
                typename tree_t,
                typename V,
                typename accumulator_t,
                typename output_t = V>
        auto propagate_sequential_and_accumulate_impl(const tree_t &tree,
                                                      array_view<const V> input,
                                                      accumulator_t &accumulator) {
            HG_TRACE();
            hg_assert_node_weights(tree, input);

            auto data_shape = std::vector<size_t>(input.shape().begin() + 1, input.shape().end());
            auto output_shape = accumulator_t::get_output_shape(data_shape);
            hg_assert((index_t) output_shape.size() == input.dimension() - 1,
                      "Input dimension does not match accumulator output dimension.");
            hg_assert(std::equal(output_shape.begin(), output_shape.end(), input.shape().begin() + 1),
                      "Input shape does not match accumulator output shape.");

            output_shape.insert(output_shape.begin(), num_vertices(tree));
            array_nd <output_t> output = array_nd<output_t>::from_shape(output_shape);

            auto input_view = make_light_axis_view<vectorial>(input);
            auto output_view = make_light_axis_view<vectorial>(output);
            auto parent_view = make_light_axis_view<vectorial>(output);

            auto aparents = parents(tree).data();
            auto acc = accumulator.template make_accumulator<vectorial>(output_view);
            // root cannot be deleted
            output_view.set_position(root(tree));
            input_view.set_position(root(tree));
            acc.set_storage(output_view);
            acc.initialize();
            acc.accumulate(input_view.begin());
            acc.finalize();

            for (auto i: root_to_leaves_iterator(tree, leaves_it::include, root_it::exclude)) {
                output_view.set_position(i);
                acc.set_storage(output_view);
                acc.initialize();

                parent_view.set_position(aparents[i]);
                acc.accumulate(parent_view.begin());

                input_view.set_position(i);
                acc.accumulate(input_view.begin());

                acc.finalize();
            }

            return output;
        };

        template<bool vectorial,
                typename tree_t,
                typename V1,
                typename T2,
                typename accumulator_t,
                typename output_t = V1>
        auto propagate_sequential_and_accumulate_impl(const tree_t &tree,
                                                      array_view<const V1> input,
                                                      const xt::xexpression<T2> &xcondition,
                                                      accumulator_t &accumulator) {
            HG_TRACE();
            auto &condition = xcondition.derived_cast();
            hg_assert_node_weights(tree, input);
            hg_assert_node_weights(tree, condition);
            hg_assert_1d_array(condition);

            auto data_shape = std::vector<size_t>(input.shape().begin() + 1, input.shape().end());
            auto output_shape = accumulator_t::get_output_shape(data_shape);
            hg_assert((index_t) output_shape.size() == input.dimension() - 1,
                      "Input dimension does not match accumulator output dimension.");
            hg_assert(std::equal(output_shape.begin(), output_shape.end(), input.shape().begin() + 1),
                      "Input shape does not match accumulator output shape.");

            output_shape.insert(output_shape.begin(), num_vertices(tree));
            array_nd <output_t> output = array_nd<output_t>::from_shape(output_shape);

            auto input_view = make_light_axis_view<vectorial>(input);
            auto output_view = make_light_axis_view<vectorial>(output);
            auto parent_view = make_light_axis_view<vectorial>(output);

            auto aparents = parents(tree).data();
            auto acc = accumulator.template make_accumulator<vectorial>(output_view);
            // root cannot be deleted
            output_view.set_position(root(tree));
            input_view.set_position(root(tree));

            if (condition(root(tree))){
                acc.set_storage(output_view);
                acc.initialize();
                acc.accumulate(input_view.begin());
                acc.finalize();
            } else {
                output_view = input_view;
            }

            for (auto i: root_to_leaves_iterator(tree, leaves_it::include, root_it::exclude)) {
                output_view.set_position(i);
                if (condition(i)){
                    acc.set_storage(output_view);
                    acc.initialize();

                    parent_view.set_position(aparents[i]);
                    acc.accumulate(parent_view.begin());

                    input_view.set_position(i);
                    acc.accumulate(input_view.begin());

                    acc.finalize();
                } else{
                    input_view.set_position(i);
                    output_view = input_view;
                }

            }

            return output;
        };

    }

    /*
     * Public entry points: array inputs that are processed row by row are obtained through
     * hg::bridge::as_view, which refers to the elements of contiguous arrays and evaluates lazy or
     * strided expressions into temporary contiguous storage.
     */

    template<typename tree_t, typename T, typename accumulator_t, typename output_t = typename T::value_type>
    auto accumulate_parallel(const tree_t &tree,
                             const xt::xexpression<T> &xinput,
                             const accumulator_t &accumulator) {
        const auto input = hg::bridge::as_view(xinput);
        if (input.view.dimension() == 1) {
            return tree_accumulator_detail::accumulate_parallel_impl<false>(tree, input.view, accumulator);
        } else {
            return tree_accumulator_detail::accumulate_parallel_impl<true>(tree, input.view, accumulator);
        }
    };


    template<typename tree_t, typename T, typename accumulator_t, typename output_t = typename T::value_type>
    auto accumulate_sequential(const tree_t &tree,
                               const xt::xexpression<T> &xvertex_data,
                               const accumulator_t &accumulator) {
        const auto vertex_data = hg::bridge::as_view(xvertex_data);

        if (vertex_data.view.dimension() == 1) {
            return tree_accumulator_detail::accumulate_sequential_impl<false>(tree, vertex_data.view, accumulator);
        } else {
            return tree_accumulator_detail::accumulate_sequential_impl<true>(tree, vertex_data.view, accumulator);
        }
    };

    template<typename tree_t, typename T1, typename T2, typename accumulator_t, typename combination_fun_t, typename output_t = typename T1::value_type>
    auto accumulate_and_combine_sequential(const tree_t &tree,
                                           const xt::xexpression<T1> &xinput,
                                           const xt::xexpression<T2> &xvertex_data,
                                           const accumulator_t &accumulator,
                                           const combination_fun_t &combine) {
        const auto input = hg::bridge::as_view(xinput);
        const auto vertex_data = hg::bridge::as_view(xvertex_data);

        if (input.view.dimension() == 1) {
            return tree_accumulator_detail::accumulate_and_combine_sequential_impl<false>(tree, input.view,
                                                                                          vertex_data.view,
                                                                                          accumulator, combine);
        } else {
            return tree_accumulator_detail::accumulate_and_combine_sequential_impl<true>(tree, input.view,
                                                                                         vertex_data.view,
                                                                                         accumulator, combine);
        }
    };

    template<typename tree_t, typename T1>
    auto propagate_parallel(const tree_t &tree,
                            const xt::xexpression<T1> &xinput) {
        const auto input = hg::bridge::as_view(xinput);

        if (input.view.dimension() == 1) {
            return tree_accumulator_detail::propagate_parallel_impl<false>(tree, input.view);
        } else {
            return tree_accumulator_detail::propagate_parallel_impl<true>(tree, input.view);
        }
    };

    template<typename tree_t, typename T1, typename T2>
    auto propagate_parallel(const tree_t &tree,
                            const xt::xexpression<T1> &xinput,
                            const xt::xexpression<T2> &xcondition) {
        const auto input = hg::bridge::as_view(xinput);

        if (input.view.dimension() == 1) {
            return tree_accumulator_detail::propagate_parallel_impl<false>(tree, input.view, xcondition);
        } else {
            return tree_accumulator_detail::propagate_parallel_impl<true>(tree, input.view, xcondition);
        }
    };

    template<typename tree_t, typename T1, typename T2>
    auto propagate_sequential(const tree_t &tree,
                              const xt::xexpression<T1> &xinput,
                              const xt::xexpression<T2> &xcondition) {
        const auto input = hg::bridge::as_view(xinput);

        if (input.view.dimension() == 1) {
            return tree_accumulator_detail::propagate_sequential_impl<false>(tree, input.view, xcondition);
        } else {
            return tree_accumulator_detail::propagate_sequential_impl<true>(tree, input.view, xcondition);
        }
    };

    template<typename tree_t, typename T, typename accumulator_t>
    auto propagate_sequential_and_accumulate(const tree_t &tree,
                                             const xt::xexpression<T> &xinput,
                                             const accumulator_t &accumulator) {
        const auto input = hg::bridge::as_view(xinput);

        if (input.view.dimension() == 1) {
            return tree_accumulator_detail::propagate_sequential_and_accumulate_impl<false>(tree, input.view, accumulator);
        } else {
            return tree_accumulator_detail::propagate_sequential_and_accumulate_impl<true>(tree, input.view, accumulator);
        }
    };

    template<typename tree_t, typename T1, typename T2, typename accumulator_t>
    auto propagate_sequential_and_accumulate(const tree_t &tree,
                                             const xt::xexpression<T1> &xinput,
                                             const accumulator_t &accumulator,
                                             const xt::xexpression<T2> &xcondition) {
        const auto input = hg::bridge::as_view(xinput);

        if (input.view.dimension() == 1) {
            return tree_accumulator_detail::propagate_sequential_and_accumulate_impl<false>(tree, input.view, xcondition, accumulator);
        } else {
            return tree_accumulator_detail::propagate_sequential_and_accumulate_impl<true>(tree, input.view, xcondition, accumulator);
        }
    };

}

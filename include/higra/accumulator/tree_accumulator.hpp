//
// Created by user on 4/15/18.
//

#pragma once

#include "../graph.hpp"
#include "accumulator.hpp"
#include "../structure/details/light_axis_view.hpp"

namespace hg {

    namespace tree_accumulator_detail {


        template<bool vectorial,
                typename tree_t,
                typename T,
                typename accumulator_t,
                typename output_t = typename T::value_type>
        auto accumulate_parallel_impl(const tree_t &tree,
                                      const xt::xexpression<T> &xinput,
                                      const accumulator_t accumulator) {
            auto &input = xinput.derived_cast();
            hg_assert(tree.num_vertices() == input.shape()[0],
                      "Size of input first dimension must be equal to the number of nodes in the tree.");
            auto data_shape = std::vector<std::size_t>(input.shape().begin() + 1, input.shape().end());
            auto output_shape = accumulator_t::get_output_shape(data_shape);
            output_shape.insert(output_shape.begin(), tree.num_vertices());

            array_nd<output_t> output = array_nd<output_t>::from_shape(output_shape);

            auto input_view = make_light_axis_view<vectorial>(input);
            auto output_view = make_light_axis_view<vectorial>(output);
            auto acc = accumulator.template make_accumulator<vectorial>(output_view);

            for (auto i: tree.iterate_on_leaves()) {
                output_view.set_position(i);
                acc.set_storage(output_view);
                acc.initialize();
                acc.finalize();
            }

            for (auto i : tree.iterate_from_leaves_to_root(leaves_it::exclude)) {
                output_view.set_position(i);
                acc.set_storage(output_view);
                acc.initialize();
                for (auto c : tree.children(i)) {
                    input_view.set_position(c);
                    acc.accumulate(input_view.begin());
                }
                acc.finalize();
            }

            return output;
        };

        template<bool vectorial,
                typename tree_t,
                typename T,
                typename accumulator_t,
                typename output_t = typename T::value_type>
        auto accumulate_sequential_impl(const tree_t &tree,
                                        const xt::xexpression<T> &xvertex_data,
                                        const accumulator_t &accumulator) {
            auto &vertex_data = xvertex_data.derived_cast();

            auto data_shape = std::vector<std::size_t>(vertex_data.shape().begin() + 1, vertex_data.shape().end());
            auto output_shape = accumulator_t::get_output_shape(data_shape);
            output_shape.insert(output_shape.begin(), tree.num_vertices());

            array_nd<output_t> output = array_nd<output_t>::from_shape(output_shape);

            auto vertex_data_view = make_light_axis_view<vectorial>(vertex_data);
            auto input_view = make_light_axis_view<vectorial>(output);
            auto output_view = make_light_axis_view<vectorial>(output);
            auto acc = accumulator.template make_accumulator<vectorial>(output_view);

            for (auto i: tree.iterate_on_leaves()) {
                output_view.set_position(i);
                vertex_data_view.set_position(i);
                output_view = vertex_data_view;
            }

            for (auto i : tree.iterate_from_leaves_to_root(leaves_it::exclude)) {
                output_view.set_position(i);
                acc.set_storage(output_view);
                acc.initialize();
                for (auto c : tree.children(i)) {
                    input_view.set_position(c);
                    acc.accumulate(input_view.begin());
                }
                acc.finalize();
            }
            return output;
        };

        template<bool vectorial,
                typename tree_t,
                typename T1,
                typename T2,
                typename accumulator_t,
                typename combination_fun_t,
                typename output_t = typename T1::value_type>
        auto accumulate_and_combine_sequential_impl(const tree_t &tree,
                                                    const xt::xexpression<T1> &xinput,
                                                    const xt::xexpression<T2> &xvertex_data,
                                                    accumulator_t &accumulator,
                                                    combination_fun_t combine) {
            auto &input = xinput.derived_cast();
            hg_assert(tree.num_vertices() == input.shape()[0],
                      "Size of input first dimension must be equal to the number of nodes in the tree.");

            auto data_shape = std::vector<std::size_t>(input.shape().begin() + 1, input.shape().end());
            auto output_shape = accumulator_t::get_output_shape(data_shape);
            output_shape.insert(output_shape.begin(), tree.num_vertices());

            array_nd<output_t> output = array_nd<output_t>::from_shape(output_shape);

            auto input_view = make_light_axis_view<vectorial>(input);
            auto inout_view = make_light_axis_view<vectorial>(output);
            auto output_view = make_light_axis_view<vectorial>(output);
            auto acc = accumulator.template make_accumulator<vectorial>(output_view);

            auto &vertex_data = xvertex_data.derived_cast();
            auto vertex_data_view = make_light_axis_view<vectorial>(vertex_data);

            for (auto i: tree.iterate_on_leaves()) {
                output_view.set_position(i);
                vertex_data_view.set_position(i);
                output_view = vertex_data_view;
            }

            for (auto i : tree.iterate_from_leaves_to_root(leaves_it::exclude)) {
                output_view.set_position(i);
                acc.set_storage(output_view);
                acc.initialize();
                for (auto c : tree.children(i)) {

                    inout_view.set_position(c);
                    acc.accumulate(inout_view.begin());
                }
                acc.finalize();
                input_view.set_position(i);
                output_view.combine(input_view, combine);
            }

            return output;
        };

        template<bool vectorial,
                typename tree_t,
                typename T1,
                typename T2,
                typename output_t = typename T1::value_type>
        auto propagate_parallel_impl(const tree_t &tree,
                                     const xt::xexpression<T1> &xinput,
                                     const xt::xexpression<T2> &xcondition) {
            auto &input = xinput.derived_cast();
            auto &condition = xcondition.derived_cast();
            hg_assert(tree.num_vertices() == input.shape()[0],
                      "Size of input first dimension must be equal to the number of nodes in the tree.");


            array_nd<output_t> output = array_nd<output_t>::from_shape(input.shape());

            auto input_view = make_light_axis_view<vectorial>(input);
            auto output_view = make_light_axis_view<vectorial>(output);

            auto parents = tree.parents().storage_begin();

            for (auto i: tree.iterate_from_root_to_leaves()) {
                if (condition(i)) {
                    input_view.set_position(parents[i]);
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
                typename T1,
                typename T2,
                typename output_t = typename T1::value_type>
        auto propagate_sequential_impl(const tree_t &tree,
                                       const xt::xexpression<T1> &xinput,
                                       const xt::xexpression<T2> &xcondition) {
            auto &input = xinput.derived_cast();

            array_nd<output_t> output = array_nd<output_t>::from_shape(input.shape());

            auto input_view = make_light_axis_view<vectorial>(input);
            auto output_view = make_light_axis_view<vectorial>(output);
            auto inout_view = make_light_axis_view<vectorial>(output);

            auto &condition = xcondition.derived_cast();
            auto parents = tree.parents().storage_begin();

            for (auto i: tree.iterate_from_root_to_leaves()) {
                output_view.set_position(i);

                if (condition(i)) {
                    inout_view.set_position(parents[i]);
                    output_view = inout_view;
                } else {
                    input_view.set_position(i);
                    output_view = input_view;
                }

            }
            return output;
        };

    }

    template<typename tree_t, typename T, typename accumulator_t, typename output_t = typename T::value_type>
    auto accumulate_parallel(const tree_t &tree,
                             const xt::xexpression<T> &xinput,
                             const accumulator_t &accumulator) {
        auto &input = xinput.derived_cast();
        if (input.dimension() == 1) {
            return tree_accumulator_detail::accumulate_parallel_impl<false>(tree, xinput, accumulator);
        } else {
            return tree_accumulator_detail::accumulate_parallel_impl<true>(tree, xinput, accumulator);
        }
    };


    template<typename tree_t, typename T, typename accumulator_t, typename output_t = typename T::value_type>
    auto accumulate_sequential(const tree_t &tree,
                               const xt::xexpression<T> &xvertex_data,
                               const accumulator_t &accumulator) {
        auto &vertex_data = xvertex_data.derived_cast();

        if (vertex_data.dimension() == 1) {
            return tree_accumulator_detail::accumulate_sequential_impl<false>(tree, xvertex_data, accumulator);
        } else {
            return tree_accumulator_detail::accumulate_sequential_impl<true>(tree, xvertex_data, accumulator);
        }
    };

    template<typename tree_t, typename T1, typename T2, typename accumulator_t, typename combination_fun_t, typename output_t = typename T1::value_type>
    auto accumulate_and_combine_sequential(const tree_t &tree,
                                           const xt::xexpression<T1> &xinput,
                                           const xt::xexpression<T2> &xvertex_data,
                                           const accumulator_t &accumulator,
                                           combination_fun_t combine) {
        auto &input = xinput.derived_cast();

        if (input.dimension() == 1) {
            return tree_accumulator_detail::accumulate_and_combine_sequential_impl<false>(tree, xinput, xvertex_data,
                                                                                          accumulator, combine);
        } else {
            return tree_accumulator_detail::accumulate_and_combine_sequential_impl<true>(tree, xinput, xvertex_data,
                                                                                         accumulator, combine);
        }

    };

    template<typename tree_t, typename T1, typename T2, typename output_t = typename T1::value_type>
    auto propagate_parallel(const tree_t &tree,
                            const xt::xexpression<T1> &xinput,
                            const xt::xexpression<T2> &xcondition) {
        auto &input = xinput.derived_cast();

        if (input.dimension() == 1) {
            return tree_accumulator_detail::propagate_parallel_impl<false>(tree, xinput, xcondition);
        } else {
            return tree_accumulator_detail::propagate_parallel_impl<true>(tree, xinput, xcondition);
        }
    };

    template<typename tree_t, typename T1, typename T2, typename output_t = typename T1::value_type>
    auto propagate_sequential(const tree_t &tree,
                              const xt::xexpression<T1> &xinput,
                              const xt::xexpression<T2> &xcondition) {
        auto &input = xinput.derived_cast();

        if (input.dimension() == 1) {
            return tree_accumulator_detail::propagate_sequential_impl<false>(tree, xinput, xcondition);
        } else {
            return tree_accumulator_detail::propagate_sequential_impl<true>(tree, xinput, xcondition);
        }
    };

/*
    template<typename tree_t, typename T, typename accumulator_t, typename output_t = typename T::value_type>
    auto accumulate_parallel_view(const tree_t &tree,
                                  const xt::xexpression<T> &xinput,
                                  accumulator_t &&accumulator) {
        auto &input = xinput.derived_cast();
        array_nd<output_t> output = xt::zeros<output_t>(input.shape());

        acc_reset(accumulator);
        auto v = acc_result(accumulator);

        for (auto i: tree.iterate_on_leaves()) {
            xt::view(output, i) = v;
        }


        for (auto i : tree.iterate_from_leaves_to_root(leaves_it::exclude)) {
            acc_reset(accumulator);
            for (auto c : tree.children(i)) {
                acc_accumulate(xt::view(input, c), accumulator);
            }
            xt::view(output, i) = acc_result(accumulator);
        }

        return output;
    };

    template<typename tree_t, typename T, typename accumulator_t, typename output_t = typename T::value_type>
    auto accumulate_sequential_view(const tree_t &tree,
                                    const xt::xexpression<T> &xvertex_data,
                                    accumulator_t &&accumulator) {
        auto &vertex_data = xvertex_data.derived_cast();
        std::vector<std::size_t> res_shape(vertex_data.shape().begin(), vertex_data.shape().end());
        res_shape[0] = num_vertices(tree);
        array_nd<output_t> output = xt::zeros<output_t>(res_shape);

        for (auto i: tree.iterate_on_leaves())
            xt::view(output, i) = xt::view(vertex_data, i);

        for (auto i : tree.iterate_from_leaves_to_root(leaves_it::exclude)) {
            acc_reset(accumulator);
            for (auto c : tree.children(i)) {
                acc_accumulate(xt::view(output, c), accumulator);
            }
            xt::view(output, i) = acc_result(accumulator);
        }
        return output;
    };

    template<typename tree_t, typename T1, typename T2, typename accumulator_t, typename combination_fun_t, typename output_t = typename T1::value_type>
    auto accumulate_and_combine_sequential_view(const tree_t &tree,
                                                const xt::xexpression<T1> &xinput,
                                                const xt::xexpression<T2> &xvertex_data,
                                                accumulator_t &&accumulator,
                                                combination_fun_t combine) {
        auto &input = xinput.derived_cast();
        auto &vertex_data = xvertex_data.derived_cast();
        std::vector<std::size_t> res_shape(vertex_data.shape().begin(), vertex_data.shape().end());
        res_shape[0] = num_vertices(tree);
        array_nd<output_t> output = xt::zeros<output_t>(res_shape);

        for (auto i: tree.iterate_on_leaves())
            xt::view(output, i) = xt::view(vertex_data, i);

        for (auto i : tree.iterate_from_leaves_to_root(leaves_it::exclude)) {
            acc_reset(accumulator);
            for (auto c : tree.children(i)) {
                acc_accumulate(xt::view(output, c), accumulator);
            }
            xt::view(output, i) = combine(acc_result(accumulator), xt::view(input, i));
        }

        return output;
    };

    template<typename tree_t, typename T1, typename T2, typename output_t = typename T1::value_type>
    auto propagate_parallel_view(const tree_t &tree,
                                 const xt::xexpression<T1> &xinput,
                                 const xt::xexpression<T2> &xcondition) {
        auto &input = xinput.derived_cast();
        auto &condition = xcondition.derived_cast();
        array_nd<output_t> output = xt::zeros<output_t>(input.shape());
        auto &parents = tree.parents();

        for (auto i: tree.iterate_from_root_to_leaves()) {
            if (condition(i)) {
                xt::view(output, i) = xt::view(input, parents(i));
            } else {
                xt::view(output, i) = xt::view(input, i);
            }
        }
        return output;
    };

    template<typename tree_t, typename T1, typename T2, typename output_t = typename T1::value_type>
    auto propagate_sequential_view(const tree_t &tree,
                                   const xt::xexpression<T1> &xinput,
                                   const xt::xexpression<T2> &xcondition) {
        auto &input = xinput.derived_cast();
        array_nd<output_t> output = xt::zeros<output_t>(input.shape());
        auto &condition = xcondition.derived_cast();
        auto &parents = tree.parents();

        for (auto i: tree.iterate_from_root_to_leaves()) {
            if (condition(i)) {
                xt::view(output, i) = xt::view(output, parents(i));
            } else {
                xt::view(output, i) = xt::view(input, i);
            }
        }
        return output;
    };*/
}
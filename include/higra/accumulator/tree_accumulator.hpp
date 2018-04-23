//
// Created by user on 4/15/18.
//

#pragma once

#include "../graph.hpp"
#include "accumulator.hpp"
#include "xtensor/xaxis_iterator.hpp"

namespace hg {

    template<typename tree_t, typename T, typename accumulator_t, typename output_t = typename T::value_type>
    auto accumulate_parallel(const tree_t &tree,
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
    auto accumulate_sequential(const tree_t &tree,
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
    auto accumulate_and_combine_sequential(const tree_t &tree,
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
    auto propagate_parallel(const tree_t &tree,
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
    auto propagate_sequential(const tree_t &tree,
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
    };
}
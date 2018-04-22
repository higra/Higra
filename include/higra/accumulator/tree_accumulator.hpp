//
// Created by user on 4/15/18.
//

#pragma once

#include "../graph.hpp"
#include "accumulator.hpp"
#include "xtensor/xaxis_iterator.hpp"

namespace hg {

    template<typename tree_t, typename T1, typename T2, typename accumulator_t>
    void accumulate_parallel(const tree_t &tree,
                             const xt::xexpression<T1> &xinput,
                             xt::xexpression<T2> &xoutput,
                             accumulator_t &&accumulator) {
        auto &input = xinput.derived_cast();
        auto &output = xoutput.derived_cast();

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
    };

    template<typename tree_t, typename T1, typename accumulator_t>
    void accumulate_sequential(const tree_t &tree,
                               xt::xexpression<T1> &xoutput,
                               accumulator_t &&accumulator) {
        auto &output = xoutput.derived_cast();

        for (auto i : tree.iterate_from_leaves_to_root(leaves_it::exclude)) {
            acc_reset(accumulator);
            for (auto c : tree.children(i)) {
                acc_accumulate(xt::view(output, c), accumulator);
            }
            xt::view(output, i) = acc_result(accumulator);
        }
    };

    template<typename tree_t, typename T1, typename T2, typename accumulator_t, typename combination_fun_t>
    void accumulate_and_combine_sequential(const tree_t &tree,
                                           const xt::xexpression<T1> &xinput,
                                           xt::xexpression<T2> &xoutput,
                                           accumulator_t &&accumulator,
                                           combination_fun_t combine) {
        auto &input = xinput.derived_cast();
        auto &output = xoutput.derived_cast();
        for (auto i : tree.iterate_from_leaves_to_root(leaves_it::exclude)) {
            acc_reset(accumulator);
            for (auto c : tree.children(i)) {
                acc_accumulate(xt::view(output, c), accumulator);
            }
            xt::view(output, i) = combine(acc_result(accumulator), xt::view(input, i));

        }
    };

    template<typename tree_t, typename T1, typename T2, typename T3>
    void propagate_parallel(const tree_t &tree,
                            const xt::xexpression<T1> &xinput,
                            xt::xexpression<T2> &xoutput,
                            const xt::xexpression<T3> &xcondition) {
        auto &input = xinput.derived_cast();
        auto &output = xoutput.derived_cast();
        auto &condition = xcondition.derived_cast();
        auto &parents = tree.parents();
        for (auto i: tree.iterate_from_root_to_leaves()) {
            if (condition(i)) {
                xt::view(output, i) = xt::view(input, parents(i));
            } else {
                xt::view(output, i) = xt::view(input, i);
            }
        }
    };

    template<typename tree_t, typename T1, typename T2, typename T3>
    void propagate_sequential(const tree_t &tree,
                              const xt::xexpression<T1> &xinput,
                              xt::xexpression<T2> &xoutput,
                              const xt::xexpression<T3> &xcondition) {
        auto &input = xinput.derived_cast();
        auto &output = xoutput.derived_cast();
        auto &condition = xcondition.derived_cast();
        auto &parents = tree.parents();
        for (auto i: tree.iterate_from_root_to_leaves()) {
            if (condition(i)) {
                xt::view(output, i) = xt::view(output, parents(i));
            } else {
                xt::view(output, i) = xt::view(input, i);
            }
        }
    };
}
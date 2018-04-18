//
// Created by user on 4/15/18.
//

#pragma once

#include "../graph.hpp"


namespace hg {

    template<typename tree_t, typename T1, typename T2, typename accumulator_t>
    void accumulate_parallel(const tree_t &tree, const xt::xexpression<T1> &xinput, xt::xexpression<T2> &xoutput,
                             accumulator_t &&accumulator) {
        auto &input = xinput.derived_cast();
        auto &output = xoutput.derived_cast();

        accumulator.reset();
        auto v = accumulator.result();
        for (auto i: tree.iterate_on_leaves())
            output(i) = v;

        for (auto i : tree.iterate_from_leaves_to_root(leaves_it::exclude)) {
            accumulator.reset();
            for (auto c : tree.children(i)) {
                accumulator.accumulate(input(c));
            }
            output(i) = accumulator.result();
        }
    };

    template<typename tree_t, typename T1, typename accumulator_t>
    void accumulate_sequential(const tree_t &tree, xt::xexpression<T1> &xoutput, accumulator_t &&accumulator) {
        auto &output = xoutput.derived_cast();

        for (auto i : tree.iterate_from_leaves_to_root(leaves_it::exclude)) {
            accumulator.reset();
            for (auto c : tree.children(i)) {
                accumulator.accumulate(output(c));
            }
            output(i) = accumulator.result();
        }
    };

    template<typename tree_t, typename T1, typename T2, typename accumulator_t, typename combination_fun_t>
    void accumulate_and_combine_sequential(const tree_t &tree, const xt::xexpression<T1> &xinput,
                                           xt::xexpression<T2> &xoutput,
                                           accumulator_t &&accumulator, combination_fun_t combine) {
        auto &input = xinput.derived_cast();
        auto &output = xoutput.derived_cast();

        for (auto i : tree.iterate_from_leaves_to_root(leaves_it::exclude)) {
            accumulator.reset();
            for (auto c : tree.children(i)) {
                accumulator.accumulate(output(c));
            }
            output(i) = combine(accumulator.result(), input(i));
        }
    };
}
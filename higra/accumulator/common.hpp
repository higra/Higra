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

#include "py_accumulators.hpp"
#include "../py_common.hpp"

/**
 * Calls fun with the accumulator object corresponding to the given enumeration value and converts its result
 * to a Python object (the result type depends on the accumulator).
 */
template<typename functor_t>
pybind11::object dispatch_accumulator(const functor_t & fun, const hg::accumulators & accumulator){
    switch (accumulator) {
        case hg::accumulators::min:
            return pybind11::cast(fun(hg::accumulator_min()));
        case hg::accumulators::max:
            return pybind11::cast(fun(hg::accumulator_max()));
        case hg::accumulators::mean:
            return pybind11::cast(fun(hg::accumulator_mean()));
        case hg::accumulators::counter:
            return pybind11::cast(fun(hg::accumulator_counter()));
        case hg::accumulators::sum:
            return pybind11::cast(fun(hg::accumulator_sum()));
        case hg::accumulators::prod:
            return pybind11::cast(fun(hg::accumulator_prod()));
        case hg::accumulators::first:
            return pybind11::cast(fun(hg::accumulator_first()));
        case hg::accumulators::last:
            return pybind11::cast(fun(hg::accumulator_last()));
        case hg::accumulators::argmin:
            return pybind11::cast(fun(hg::accumulator_argmin()));
        case hg::accumulators::argmax:
            return pybind11::cast(fun(hg::accumulator_argmax()));
    }
    hg::unreachable();
}

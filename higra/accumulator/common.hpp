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

template<typename functor_t>
auto dispatch_accumulator(const functor_t & fun, const hg::accumulators & accumulator){
    switch (accumulator) {
        case hg::accumulators::min:
            return fun(hg::accumulator_min());
        case hg::accumulators::max:
            return fun(hg::accumulator_max());
        case hg::accumulators::mean:
            return fun(hg::accumulator_mean());
        case hg::accumulators::counter:
            return fun(hg::accumulator_counter());
        case hg::accumulators::sum:
            return fun(hg::accumulator_sum());
        case hg::accumulators::prod:
            return fun(hg::accumulator_prod());
        case hg::accumulators::first:
            return fun(hg::accumulator_first());
        case hg::accumulators::last:
            return fun(hg::accumulator_last());
    }
}

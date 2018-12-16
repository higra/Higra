/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_accumulators.hpp"
#include "../py_common.hpp"
#include "higra/accumulator/accumulator.hpp"

namespace py = pybind11;

void py_init_accumulators(pybind11::module &m) {

    // only exposes enumeration
    py::enum_<hg::accumulators>(m, "Accumulators")
            .value("min", hg::accumulators::min)
            .value("max", hg::accumulators::max)
            .value("mean", hg::accumulators::mean)
            .value("counter", hg::accumulators::counter)
            .value("sum", hg::accumulators::sum)
            .value("prod", hg::accumulators::prod)
            .value("first", hg::accumulators::first)
            .value("last", hg::accumulators::last)
            .export_values();
}
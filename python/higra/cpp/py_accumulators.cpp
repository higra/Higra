//
// Created by user on 4/17/18.
//

#include "py_accumulators.hpp"
#include "py_common.hpp"
#include "higra/accumulator/accumulator.hpp"

namespace py = pybind11;

void py_init_accumulators(pybind11::module &m) {

    // only exposes enumeration
    py::enum_<hg::accumulators>(m, "Accumulators")
            .value("min", hg::accumulators::min)
            .value("max", hg::accumulators::max)
            .value("counter", hg::accumulators::counter)
            .value("sum", hg::accumulators::sum)
            .value("prod", hg::accumulators::prod);
}
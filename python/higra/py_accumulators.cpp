//
// Created by user on 4/17/18.
//

#include "py_accumulators.hpp"
#include "pybind11/stl.h"
#include "utils.hpp"
#include "accumulator.hpp"

namespace py = pybind11;

void py_init_accumulators(pybind11::module &m) {

    // only exposes enumeration
    py::enum_<hg::accumulators>(m, "Accumulators")
#define DEF(rawXKCD, dataXKCD, acc_name) \
        .value(HG_XSTR(acc_name), hg::accumulators::acc_name)
            HG_FOREACH(DEF, HG_ACCUMULATORS);
#undef DEF

}
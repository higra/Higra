//
// Created by perretb on 28/05/18.
//

#include "py_log.hpp"
#include "py_common.hpp"
#include "higra/detail/log.hpp"

void py_init_log(pybind11::module &m) {
    m.def("set_trace", [](bool enabled) { hg::trace::enabled() = enabled; },
          "Define if function call tracing is enabled.",
          pybind11::arg("enabled"));

    m.def("get_trace", []() { return hg::trace::enabled(); },
          "Get the state of function call tracing.");
}

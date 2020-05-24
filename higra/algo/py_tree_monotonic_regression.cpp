/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_tree_monotonic_regression.hpp"
#include "higra/algo/tree_monotonic_regression.hpp"
#include "../py_common.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

template<typename T>
using pyarray = xt::pyarray<T>;

namespace py = pybind11;
using namespace hg;

struct def_tree_monotonic_regression {
    template<typename type, typename C>
    static
    void def(C &m, const char *doc) {
        m.def("_tree_monotonic_regression",
              [](const hg::tree &tree,
                 const pyarray<type> &altitudes,
                 const std::string mode,
                 const pyarray<double> &weights = {}) {
                  return hg::tree_monotonic_regression(tree, altitudes, weights, mode);
              },
              doc,
              py::arg("tree"),
              py::arg("altitudes"),
              py::arg("mode"),
              py::arg("weights") = hg::array_1d<double>());
    }
};

void py_init_tree_monotonic_regression(pybind11::module &m) {
    xt::import_numpy();

    add_type_overloads<def_tree_monotonic_regression, HG_TEMPLATE_NUMERIC_TYPES>(m, "");
}
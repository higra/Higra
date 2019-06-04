/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "py_dendrogram_purity.hpp"
#include "../py_common.hpp"
#include "higra/assessment/dendrogram_purity.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

using namespace hg;
namespace py = pybind11;

struct def_dendrogram_purity {
    template<typename value_type, typename C>
    static
    void def(C &c, const char *doc) {
        c.def("_dendrogram_purity",
              [](const tree &t,
                 const xt::pyarray<value_type> &leaf_labels) {
                  return dendrogram_purity(t, leaf_labels);
              },
              doc,
              py::arg("tree"),
              py::arg("leaf_labels"));
    }
};

void py_init_dendrogram_purity(pybind11::module &m) {
    xt::import_numpy();

    add_type_overloads<def_dendrogram_purity, HG_TEMPLATE_INTEGRAL_TYPES>(m, "");
}
